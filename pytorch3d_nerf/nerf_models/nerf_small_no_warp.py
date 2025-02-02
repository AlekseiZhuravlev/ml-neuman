import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import pytorch3d
import sys
import sys
import time
import torch
import torch
from IPython import display
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from tqdm.notebook import tqdm


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        # TODO - removed this
        # self.density_layer[0].bias.data[0] = -1.5
        self.density_layer[0].bias.data[0] = -1.0

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        # Normalize the ray_directions to unit l2 norm.

        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding.forward(
            rays_directions_normed
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
            self,
            ray_points,
            ray_directions,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding.forward(
            ray_points
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.
        ray_densities = self._get_densities(features)
        ray_colors = self._get_colors(
            features,
            ray_directions,
        )
        return ray_densities, ray_colors


    def batched_forward(
            self,
            ray_points,
            ray_directions,
            n_batches: int = 16,
    ):
        """
        Args:
            ray_points: torch.Size([1, 8192, 1, 32, 3])
            ray_directions: torch.Size([1, 8192, 1, 32, 3])
            n_batches: int = 16
        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
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


class NeuralRadianceFieldNoDir(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        # TODO - removed this
        # self.density_layer[0].bias.data[0] = -1.5
        self.density_layer[0].bias.data[0] = -1.0

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        # Normalize the ray_directions to unit l2 norm.

        return self.color_layer(features)

    def forward(
            self,
            ray_points,
            ray_directions,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding.forward(
            ray_points
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.
        ray_densities = self._get_densities(features)
        ray_colors = self._get_colors(
            features,
            ray_directions,
        )
        return ray_densities, ray_colors


    def batched_forward(
            self,
            ray_points,
            ray_directions,
            n_batches: int = 16,
    ):
        """
        Args:
            ray_points: torch.Size([1, 8192, 1, 32, 3])
            ray_directions: torch.Size([1, 8192, 1, 32, 3])
            n_batches: int = 16
        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
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



class OffsetNetwork(torch.nn.Module):
    def __init__(self, input_dim, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim_pts = n_harmonic_functions * 2 * input_dim
        embedding_dim_dirs = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim_pts, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.points_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim_dirs, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
        )

    def _get_offset(self, features, rays_directions):
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding.forward(
            rays_directions_normed
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        points_layer_input = torch.cat(
            (features, rays_embedding),
            dim=-1
        )
        return self.points_layer(points_layer_input)

    def forward(
            self,
            ray_points,
            ray_directions,
    ):
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding.forward(
            ray_points
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.
        ray_offset = self._get_offset(
            features,
            ray_directions,
        )
        return ray_offset


    def batched_forward(
            self,
            ray_points,
            ray_directions,
            n_batches: int = 16,
    ):
        batches_ray_points = torch.chunk(ray_points, chunks=n_batches, dim=1)
        batches_ray_directions = torch.chunk(ray_directions, chunks=n_batches, dim=1)

        # For each batch, execute the standard forward pass and concatenate
        rays_offsets = torch.tensor([], device=ray_points.device)

        for batch_idx in range(len(batches_ray_points)):
            rays_offsets_batch = self.forward(
                ray_points=batches_ray_points[batch_idx],
                ray_directions=batches_ray_directions[batch_idx],
            )
            rays_offsets = torch.cat([rays_offsets, rays_offsets_batch], dim=1)

        return rays_offsets



class OffsetNetworkNoDir(torch.nn.Module):
    def __init__(self, input_dim, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim_pts = n_harmonic_functions * 2 * input_dim

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim_pts, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.points_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
        )

    def _get_offset(self, features):
        return self.points_layer(features)

    def forward(
            self,
            ray_points,
            ray_directions,
    ):
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding.forward(
            ray_points
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.
        ray_offset = self._get_offset(
            features,
        )
        return ray_offset


    def batched_forward(
            self,
            ray_points,
            ray_directions,
            n_batches: int = 16,
    ):
        batches_ray_points = torch.chunk(ray_points, chunks=n_batches, dim=1)
        batches_ray_directions = torch.chunk(ray_directions, chunks=n_batches, dim=1)

        # For each batch, execute the standard forward pass and concatenate
        rays_offsets = torch.tensor([], device=ray_points.device)

        for batch_idx in range(len(batches_ray_points)):
            rays_offsets_batch = self.forward(
                ray_points=batches_ray_points[batch_idx],
                ray_directions=batches_ray_directions[batch_idx],
            )
            rays_offsets = torch.cat([rays_offsets, rays_offsets_batch], dim=1)

        return rays_offsets


class NeuralRadianceFieldPoseCond(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim + 48, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        # TODO - removed this
        # self.density_layer[0].bias.data[0] = -1.5
        self.density_layer[0].bias.data[0] = -1.0

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        # Normalize the ray_directions to unit l2 norm.

        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding.forward(
            rays_directions_normed
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
            self,
            ray_points,
            ray_directions,
            mano_pose
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding.forward(
            ray_points
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # add mano pose to each ray point
        # ray_points_can.shape = [1, 548, 337, 3] mano_pose.shape = [1, 48]

        # print('mano_pose', mano_pose.shape)
        # print('embeds', embeds.shape)
        # # concat mano_pose to ray_points_can
        mano_pose = mano_pose.reshape(1, 1, 1, 1, -1).repeat(
            embeds.shape[0],
            embeds.shape[1],
            embeds.shape[2],
            embeds.shape[3],
            1
        )
        # print('mano_pose', mano_pose.shape)
        embeds = torch.cat([embeds, mano_pose], dim=-1)
        # print('ray_points_can', embeds.shape)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.
        ray_densities = self._get_densities(features)
        ray_colors = self._get_colors(
            features,
            ray_directions,
        )
        return ray_densities, ray_colors


    def batched_forward(
            self,
            ray_points,
            ray_directions,
            mano_pose,
            n_batches: int = 16,
    ):
        """
        Args:
            ray_points: torch.Size([1, 8192, 1, 32, 3])
            ray_directions: torch.Size([1, 8192, 1, 32, 3])
            n_batches: int = 16
        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        batches_ray_points = torch.chunk(ray_points, chunks=n_batches, dim=1)
        batches_ray_directions = torch.chunk(ray_directions, chunks=n_batches, dim=1)

        # For each batch, execute the standard forward pass and concatenate
        rays_densities = torch.tensor([], device=ray_points.device)
        rays_colors = torch.tensor([], device=ray_points.device)

        for batch_idx in range(len(batches_ray_points)):
            rays_densities_batch, rays_colors_batch = self.forward(
                ray_points=batches_ray_points[batch_idx],
                ray_directions=batches_ray_directions[batch_idx],
                mano_pose=mano_pose
            )
            rays_densities = torch.cat([rays_densities, rays_densities_batch], dim=1)
            rays_colors = torch.cat([rays_colors, rays_colors_batch], dim=1)

        return rays_densities, rays_colors