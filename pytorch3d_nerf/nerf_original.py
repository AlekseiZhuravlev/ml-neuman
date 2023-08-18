from typing import Tuple

import torch
from pytorch3d.common.linear_with_repeat import LinearWithRepeat
from pytorch3d.renderer import HarmonicEmbedding, ray_bundle_to_ray_points, RayBundle
from mlp_with_skips import MLPWithInputSkips
from pytorch3d_nerf.warping import warp_points


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    return
    # torch.nn.init.xavier_uniform_(linear.weight.data)


class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 60,
        n_harmonic_functions_dir: int = 24,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int, ...] = (5,),
        use_multiple_streams: bool = False,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
            use_multiple_streams: Whether density and color should be calculated on
                separate CUDA streams.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(
            n_layers_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
        )

        self.intermediate_linear = torch.nn.Linear(
            n_hidden_neurons_xyz, n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        # TODO changed
        # self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons_xyz, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )
        # _xavier_init(self.density_layer)
        _xavier_init(self.density_layer[0])

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer[0].bias.data[:] = -1.0  # fixme: Sometimes this is not enough

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        self.use_multiple_streams = use_multiple_streams

    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

        # deltas = torch.cat(
        #     (
        #         depth_values[..., 1:] - depth_values[..., :-1],
        #         1e10 * torch.ones_like(depth_values[..., :1]),
        #     ),
        #     dim=-1,
        # )[..., None]
        # if density_noise_std > 0.0:
        #     raw_densities = (
        #         raw_densities + torch.randn_like(raw_densities) * density_noise_std
        #     )
        # densities = 1 - (-deltas * torch.relu(raw_densities)).exp()
        # return densities

    def _get_colors(
        self, features: torch.Tensor, rays_directions: torch.Tensor, warp_rays
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        if warp_rays:
            rays_embedding_expand = rays_embedding
        else:
            spatial_size = features.shape[:-1]
            rays_embedding_expand = rays_embedding[..., None, :].expand(
                *spatial_size, rays_embedding.shape[-1]
            )

        # color_layer_input = torch.cat(
        #     (features, rays_embedding_expand),
        #     dim=-1
        # )
        # return self.color_layer(color_layer_input)
        return self.color_layer((self.intermediate_linear(features), rays_embedding_expand))

    def forward(
        self,
        ray_bundle: RayBundle,
        vertices,
        Ts,
        warp_rays,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        if warp_rays:
            # Warp the rays to the canonical view.
            rays_points_world, ray_directions = warp_points.warp_points(
                rays_points_world,
                vertices,
                Ts,
            )
        else:
            ray_directions = ray_bundle.directions

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities = self._get_densities(
            features, ray_bundle.lengths, density_noise_std
        )
        rays_colors = self._get_colors(features, ray_directions, warp_rays)

        return rays_densities, rays_colors

    def batched_forward(self,
                        ray_bundle,
                        n_batches: int = 16,
                        **kwargs,
                        ):
        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                ),
                **kwargs,
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors

    def forward_points(
            self,
            points,
            directions,
            vertices,
            Ts,
            warp_rays,
            **kwargs,
    ):
        if warp_rays:
            # Warp the rays to the canonical view.
            rays_points_world, ray_directions = warp_points.warp_points(
                points,
                vertices,
                Ts,
            )
        else:
            ray_directions = directions
            rays_points_world = points

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities = self._get_densities(
            features, ray_bundle.lengths, density_noise_std
        )
        # TODO this should warp rays as well
        rays_colors = self._get_colors(features, ray_directions, warp_rays)

        return rays_densities, rays_colors
