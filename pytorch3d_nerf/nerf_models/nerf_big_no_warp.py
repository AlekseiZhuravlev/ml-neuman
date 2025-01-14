from typing import Tuple

import torch
from pytorch3d.common.linear_with_repeat import LinearWithRepeat
from pytorch3d.renderer import HarmonicEmbedding, ray_bundle_to_ray_points, RayBundle


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
        self.density_layer[0].bias.data[:] = -1.0

        self.color_layer = torch.nn.Sequential(
            # LinearWithRepeat(
            #     n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            # ),
            torch.nn.Linear(n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        self.use_multiple_streams = use_multiple_streams

    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor=None,
        density_noise_std: float=0.0,
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
        self, features: torch.Tensor, rays_directions: torch.Tensor
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

        color_layer_input = torch.cat(
            (features, rays_embedding),
            dim=-1
        )
        return self.color_layer(color_layer_input)

        # return self.color_layer((self.intermediate_linear(features), rays_embedding))


    def forward(
        self,
        ray_points,
        ray_directions,
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
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(ray_points)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        ray_densities = self._get_densities(features)
        ray_colors = self._get_colors(
            features,
            ray_directions,
        )
        return ray_densities, ray_colors

    def batched_forward(self,
                        ray_points,
                        ray_directions,
                        n_batches: int = 16,
                        **kwargs,
                        ):
        batches_ray_points = torch.chunk(ray_points, chunks=n_batches, dim=1)
        batches_ray_directions = torch.chunk(ray_directions, chunks=n_batches, dim=1)

        # For each batch, execute the standard forward pass and concatenate
        rays_densities = torch.tensor([], device=ray_points.device)
        rays_colors = torch.tensor([], device=ray_points.device)

        for batch_idx in range(len(batches_ray_points)):
            rays_densities_batch, rays_colors_batch = self.forward(
                ray_points=batches_ray_points[batch_idx],
                ray_directions=batches_ray_directions[batch_idx],
            )
            rays_densities = torch.cat([rays_densities, rays_densities_batch], dim=1)
            rays_colors = torch.cat([rays_colors, rays_colors_batch], dim=1)

        return rays_densities, rays_colors


    def batched_forward_with_cpu(self,
                        ray_points,
                        ray_directions,
                        n_batches: int = 16,
                        **kwargs,
                        ):

        assert ray_points.device == ray_directions.device == torch.device('cpu')

        batches_ray_points = torch.chunk(ray_points, chunks=n_batches, dim=1)
        batches_ray_directions = torch.chunk(ray_directions, chunks=n_batches, dim=1)

        # For each batch, execute the standard forward pass and concatenate
        rays_densities = torch.tensor([], device='cpu')
        rays_colors = torch.tensor([], device='cpu')

        for batch_idx in range(len(batches_ray_points)):
            rays_pts_batch = batches_ray_points[batch_idx].to('cuda')
            rays_dirs_batch = batches_ray_directions[batch_idx].to('cuda')

            rays_densities_batch, rays_colors_batch = self.forward(
                ray_points=rays_pts_batch,
                ray_directions=rays_dirs_batch,
            )
            rays_densities = torch.cat([
                rays_densities,
                rays_densities_batch.to('cpu')
            ], dim=1)
            rays_colors = torch.cat([
                rays_colors,
                rays_colors_batch.to('cpu')
            ], dim=1)

        return rays_densities, rays_colors


    def forward_points(
            self,
            rays_points_world,
            ray_directions,
            **kwargs,
    ):
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities = self._get_densities(
            features, torch.Tensor(0), 0.0
        )
        rays_colors = self._get_colors(features, ray_directions)

        return rays_densities, rays_colors


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int, ...] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y