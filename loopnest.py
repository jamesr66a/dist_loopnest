import torch
import torch.fx

from itertools import product
import operator

@torch.fx.wrap
def program_id(axis):
  pass

@torch.fx.wrap
def num_programs(axis):
  pass

@torch.fx.wrap
def broadcast(input, other):
  pass

@torch.fx.wrap
def broadcast_to(input, shape):
  pass

@torch.fx.wrap
def load(ptr):
  pass

@torch.fx.wrap
def store(ptr, val):
  pass


allowed_fns = {
  program_id, num_programs, torch.arange, torch.zeros, broadcast, broadcast_to,
  torch.cat, torch.reshape, torch.dot, load, store, torch.where, torch.div,
  torch.floor_divide, torch.exp, torch.log, torch.cos, torch.sin, torch.sqrt,
  torch.max, torch.argmax, torch.min, torch.argmin, torch.sum, torch.abs,
  torch.minimum, torch.maximum, torch.sigmoid, torch.softmax, torch.ravel,
  torch.zeros_like, operator.add, operator.sub, operator.mul, operator.truediv,
  operator.floordiv
}

# TODO: limit types and have special proxies
class TiledProgramTracer(torch.fx.Tracer):
  def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):

    if kind == 'placeholder':
      pass
    elif kind == 'call_method':
      # TODO
      raise ValueError('Method calles currently not allowed')
    elif kind == 'call_module':
      raise ValueError('Module calls not allowed in tiled program tracing!')
    elif kind == 'call_function':
      assert target in allowed_fns, \
        f'{torch.typename(target)} not allowed in tiled programming model'
    elif kind == 'get_attr':
      raise ValueError('Attribute fetches not allowed in tiled program tracing!')
    elif kind == 'output':
      pass
    else:
      pass

    return super().create_node(kind, target, args, kwargs, name, type_expr)

class IterationSpaceCallable(torch.fx.GraphModule):
  def __init__(self, axes, root, graph, class_name = 'graph_module'):
    super().__init__(root, graph, class_name)

    self.axes = axes

  def __call__(self, *args):
    # TODO: broadcasting
    assert all(isinstance(arg, torch.Tensor) for arg in args)
    assert all(arg.is_contiguous() for arg in args)
    assert all(len(self.axes) == arg.ndim for arg in args)

    # TODO: analyze outputs
    # TODO: infer device
    output = torch.empty(*self.axes, device='cuda')
    for idxs in product(*(range(axis) for axis in self.axes)):
      iter_args = [arg[idxs] for arg in args]

      output[idxs] = super().__call__(*iter_args)

    return output

def compile_loopnest(axes):

  def inner(func):
    tracer = TiledProgramTracer()
    graph = tracer.trace(func)
    return IterationSpaceCallable(axes, tracer.root, graph)

  return inner


# !!!!!! TODO: swtich to tile-based model
@compile_loopnest([3, 4])
def foobarbaz(x):
  return x * 2.0


print(foobarbaz.graph)
x = torch.randn(3, 4, device='cuda')
test_out = foobarbaz(x)
ref_out = x * 2.0

torch.testing.assert_close(test_out, ref_out)