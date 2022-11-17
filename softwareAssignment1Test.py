# Import your class (it should be in same directly as this file)
import kruskalClass

# Import numpy
import numpy as np

# Instantiate an object for your class.
obj = kruskalClass.kruskalClass()

# Create a test matrix
A = np.array([[0, 8, 0, 3], [0, 0, 2, 5], [0, 0, 0, 6], [0, 0, 0, 0]])

# Use code to generate a MST
T = obj.findMinimumSpanningTree(A)

# Print the MST
print(T)
print(type(T))

# Now we will test union-find code. Make a union-find structure for a graph with 5 nodes.
# Note that in 'u' each node has a 2-dimensional numpy array associated with it
# The first entry is a pointer.  If the pointer and the key are the same, then
# ...the name of the set to which the node belongs is itself.
# The second entry in each numpy array is a count of the number of pointers that point
# to the given node.  This is not part of the API, but I found it useful.
n = 5
u = obj.makeUnionFind(n)
print(u)

# If we run the 'find' it returns the index
# that was provided as input
s1 = obj.find(u, 2)
print(s1)
s2 = obj.find(u, 4)
print(s2)

# Now we can try doing some union operations
# Combine the sets for nodes 0 and 1
u1 = obj.union(u, obj.find(u, 0), obj.find(u, 1))
print(obj.find(u1, 0))
print(u1)
u2 = obj.union(u, obj.find(u1, 0), obj.find(u1, 2))
print(u2)

# Notice that the set '2' takes the name of the larger set,
# which is composed of {0,1} from the first merging operation.
# When doing the second union operation, your code should always give node '2'
# the name obj.find(0) (which may be '1' or '0' depending upon your implementation)
# because it is the larger set
