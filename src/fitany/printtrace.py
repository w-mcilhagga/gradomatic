


# ## Displaying computational graphs.
# 
# `printnode` will give a nested function-call trace of the computational network rooted in the node.
# 
# `graph_fwd` gives a graphviz representation of the computational graph, suitable for forward mode.


from graphviz import Digraph
import inspect
import numbers

def printNode(n, show_id=False):
    print(node2str(n, show_id=show_id))

def node2str(n, prefix='', show_id=False):
    if type(n) is VarNode:
        return prefix+nodename(n, show_id=show_id)
    if type(n) is OpNode:
        return ( prefix+nodename(n, show_id=show_id)+
                '(\n'+',\n'.join([node2str(a, prefix=prefix+'   ', show_id=show_id) for a in n.args])+'\n'+prefix+')')
    if type(n) is str:
        return prefix+nodename(n, show_id=show_id)
    if type(n) is np.ndarray:
        return prefix+nodename(n, show_id=show_id)
    return prefix+nodename(n, show_id=show_id)

def nodename(n, idlist={}, show_id=True):
    if show_id and type(n) is not str:
        try:
            idno = idlist[id(n)]
        except:
            idlist[id(n)] = len(idlist)
            idno = idlist[id(n)]
        idstr = '['+str(idno)+']' 
    else:
        idstr = ''
    if type(n) is VarNode:
        return 'VN('+nodename(n.value,{},show_id=False)+')'+idstr
    if type(n) is OpNode:
        return n.fn.__name__ + idstr
    if type(n) is str:
        return "'"+n+"'" +idstr
    if type(n) is np.ndarray:
        return 'array'+str(n.shape)+idstr
    return str(n)+idstr


def nodeid(n, idlist, names, dot=None):
    # works out an id for the node, or returns the existing id
    if np.isscalar(n):
        # this bit doesn't seem to be called
        node = dict(name=str(np.random.rand()), rendered=False)
        dot.node(node['name'], nodelabel(n, names))
        return node
    try:
        # is the node in the idlist?
        return idlist[id(n)]
    except:
        # no, so put it there. The name is a unique number,
        # 
        idlist[id(n)] = dict(name=str(len(idlist)), rendered=False)
        return idlist[id(n)]

def nodelabel(n, names):
    # works out a label for the node
    if type(n) is VarNode:
        return 'VN('+nodelabel(n.value, names)+')'
    if type(n) is OpNode:
        if n.fn == np.einsum:
            return 'einsum: '+n.args[0] # the einsum script
        else:
            return n.fn.__name__ 
    if type(n) is str:
        return f"'{n}'"
    if type(n) is np.ndarray:
        return names.get(id(n), 'array'+str(n.shape))
    if isinstance(n, numbers.Number):
        return str(n)
    if inspect.isclass(type(n)):
        return type(n).__name__+' object'

    return str(n)

def graph_fwd(n, names={}, dot=None, idlist=None):
    # creates a dot graph for the node
    if dot is None:
        # initialize
        dot = Digraph(strict=False)
        dot.attr(size='10,10')
        idlist={}
    if np.isscalar(n):
        # not part of the graph, unsure why
        return
    if idlist.get(id(n)) is None:
        # the node needs to be created.
        dot.node(nodeid(n, idlist, names, dot)['name'], nodelabel(n, names))
        # opnodes need their args to be drawn
        if type(n) is OpNode:
            # einsum args drop the script
            args = n.args if n.fn!=np.einsum else n.args[1:]
            for i, a in enumerate(args):
                # draw the arg
                graph_fwd(a, names, dot, idlist)
                # create an edge from this node to the arg
                label = str(i)
                dot.edge(nodeid(n, idlist, names, dot)['name'], nodeid(a, idlist, names, dot)['name'], label=label)
    # return the dot
    return dot


def graph_rev(n, names={}, dot=None, idlist=None):
    # creates a dot graph for the node
    if dot is None:
        dot = Digraph(strict=False)
        dot.attr(size='10,10')
        idlist={}
    if np.isscalar(n):
        return
    if idlist.get(id(n)) is None: 
        dot.node(nodeid(n, idlist, names, dot)['name'], nodelabel(n, names))
        if n in n.root.backrefs:
            for a, i in n.root.backrefs[n]:
                graph_rev(a, names, dot, idlist)
                label = str(i) 
                dot.edge(nodeid(n, idlist, names, dot)['name'], nodeid(a, idlist, names, dot)['name'], label=label)
    return dot














