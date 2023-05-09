"""gate.py - Generate Path Gates"""
import numpy as np

__all__ = ['nearest_neighbor_gate']

def nearest_neighbor_gate(boundary: list[np.array], scale_factor: float, 
                          spacing: float = 10.0, loop: bool = False) \
                         -> tuple[list[np.array], np.array]:
    """Generate path gates with given spacing"""
    spacing = spacing / scale_factor

    # Refine boundaries to desired spacing for gate posts
    post = []
    for bndry in boundary:
        if loop:
            bndry = np.append(bndry, bndry[:,[0]], axis=1)

        dist = np.linalg.norm(np.diff(bndry), axis=0)
        dist = np.flip(dist, axis=0)

        l = dist.shape[0]
        for i, d in enumerate(dist):
            j = l - i
            if (n := np.ceil(d / spacing)) > 1:
                p0, p1 = bndry[:,j-1], bndry[:,j]

                for s in np.linspace(1,0,int(n))[1:-1]:
                    bndry = np.insert(bndry, j, p0 + s*(p1 - p0), axis=1) 

        post.append(bndry)

    # Establish gates via nearest neighbor
    nn = lambda i, k: np.argmin(np.linalg.norm(post[np.mod(i+1,2)].T - post[i][:,k].T, axis=1)) 
    
    s = 1 if loop is True else 0
    gate = np.array([[k0, nn(0,k0)] for k0 in range(post[0].shape[-1]-s)], dtype=int).T
    for k1 in reversed(range(post[1].shape[-1]-s)):
        k0 = nn(1,k1)
        if k1 < gate[1,k0]: # Insert before existing gate
            gate = np.insert(gate, k0, np.array([[k0, k1]], dtype=int), axis=1)
        elif k1 > gate[1,k0]:
            if k0 >= gate.shape[1]-1:
                gate = np.append(gate, np.array([[k0], [k1]], dtype=int), axis=1)
            else:
                gate = np.insert(gate, k0+1, np.array([[k0, k1]], dtype=int), axis=1)

    crossed = np.argwhere(np.any(np.diff(gate) < 0, axis=0))
    gate = np.delete(gate, crossed, axis=1)

    return post, gate