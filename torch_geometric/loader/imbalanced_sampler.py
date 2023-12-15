from typing import List, Optional, Union
from torch_geometric.typing import InputNodes

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData, Dataset, InMemoryDataset


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).

    **Graph-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, ImbalancedSampler

        sampler = ImbalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    You can pass a heterogeneous graph and use the `input_nodes` argument to
    indicate which node type to sample upon:

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        input_nodes = ('target', hetero_data['target'].train_mask)
        sampler = ImbalancedSampler(hetero_data, input_nodes=input_nodes)
        loader = NeighborLoader(data, input_nodes=input_nodes),
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    You can also pass in the class labels directly as a :class:`torch.Tensor`:

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data.y)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset/Data/HeteroData/Tensor/List[Data]/List[HeteroData]):
            The dataset or class distribution from which to sample the data,
            given as one of :class:`~torch_geometric.data.Dataset`,
            :class:`~torch_geometric.data.Data` (or a list of such),
            :class:`~torch_geometric.data.HeteroData` (or a list of such),
            or :class:`torch.Tensor` object.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes that are used by the corresponding loader, *e.g.*,
            by :class:`~torch_geometric.loader.NeighborLoader`.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as either a tuple that
            holds the node type and node indices, or just the node type to use
            all nodes of this type.
            This argument should only be set for node-level loaders and does
            not have any effect when operating on a set of graphs as given by
            :class:`~torch_geometric.data.Dataset`. (default: :obj:`None`)
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to :obj:`None`, will sample as many elements as there
            exists in the underlying data. (default: :obj:`None`)
    """
    def __init__(
        self,
        dataset: Union[Dataset, Data, HeteroData, List[Data], List[HeteroData], Tensor],
        input_nodes: InputNodes = None,
        num_samples: Optional[int] = None,
    ):
        if isinstance(dataset, Data):
            y = dataset.y.view(-1)
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, Tensor):
            y = dataset.view(-1)
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y.view(-1)
            assert len(dataset) == y.numel()

        elif isinstance(dataset, HeteroData):
            node_type, input_nodes = (input_nodes, None) if isinstance(input_nodes, str) else input_nodes
            y = dataset[node_type].y.view(-1)
            assert dataset[node_type].num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        else:
            if isinstance(dataset[0], HeteroData):
                assert isinstance(input_nodes, str) or input_nodes[1] is None
                node_type = input_nodes if isinstance(input_nodes, str) else input_nodes[0]
                ys = [data[node_type].y for data in dataset]
            else:
                ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1. / y.bincount()
        weight = class_weight[y]

        super().__init__(weight, num_samples, replacement=True)
