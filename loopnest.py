import torch
import torch.fx

from functools import wraps
from itertools import product

class IterationSpaceCallable(torch.fx.GraphModule):
  def __init__(self, axes, root, graph, class_name = 'graph_module'):
    super().__init__(root, graph, class_name)

    self.axes = axes

  def __call__(self, *args):
    # TODO: broadcasting
    assert all(isinstance(arg, torch.Tensor) for arg in args)
    assert all(arg.is_contiguous() for arg in args)
    assert all(len(self.axes) == arg.ndim for arg in args)

    # TODO: multiple outputs
    output = torch.empty(*self.axes)
    for idxs in product(*(range(axis) for axis in self.axes)):
      iter_args = [arg[idxs] for arg in args]

      output[idxs] = super().__call__(*iter_args)

    return output

def compile_loopnest(axes):

  def inner(func):
    tracer = torch.fx.Tracer()
    graph = tracer.trace(func)
    return IterationSpaceCallable(axes, tracer.root, graph)

  return inner


# !!!!!! TODO: swtich to tile-based model
@compile_loopnest([3, 4])
def foobarbaz(x):
  return x * 2.0


print(foobarbaz.graph)
x = torch.randn(3, 4)
test_out = foobarbaz(x)
ref_out = x * 2.0

torch.testing.assert_allclose(test_out, ref_out)