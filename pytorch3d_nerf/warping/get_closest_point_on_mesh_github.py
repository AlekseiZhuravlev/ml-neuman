import torch

pmkeys = ['a','b','c','h', 'delta'] # values that can be precomputed
            # values that should be computed on the fly
            # 'd','e','f','sbar','tbar','g','k','bd','ce','be','ad']
class ProjectMesh2Point:
    def __init__(self, mesh):
        # Note. edge is oriented in right-handed direction,

        self.E0 = mesh.verts_packed()[mesh.edges_packed()][:, 0, :]
        self.E1 = mesh.verts_packed()[mesh.edges_packed()][:, 1, :]
        self.B = mesh.verts_packed()[mesh.faces_packed()][:, 0, :]

        # self.E0 = mesh.edges[:,0].squeeze()
        # self.E1 = -mesh.edges[:,1].squeeze()
        # self.B = mesh.triangles[:,0].squeeze()
        for key in pmkeys:
            setattr(self, key, [])

        # Values that can be precomputed
        self.a = torch.sum(self.E0 * self.E0, dim=-1) # inner product
        self.b = torch.sum(self.E0 * self.E1, dim=-1)
        self.c = torch.sum(self.E1 * self.E1, dim=-1)
        self.delta = self.a * self.c - self.b**2
        self.h = self.a - 2*self.b + self.c

    def __call__(self, query, index):
        """
        The projected point is computed by
        Pout = B + sout*E0 + tout*E1
        query : Px3 vector
        index : Px1 indices for updating vars
        """
        # Update projection variables according to input queries
        # variables of "d,e,f,sbar,tbar,g,k,bd,ce,be,ad" are updated

        # Get vars in indices
        B, E0, E1 = self.B[index,:], self.E0[index,:], self.E1[index,:]
        a, b, c, h = self.a[index], self.b[index], self.c[index], self.h[index]
        delta = self.delta[index]
        D = B - query
        d = torch.sum(E0*D, dim=-1)
        e = torch.sum(E1*D, dim=-1)
        # f = torch.sum(D*D, dim=-1)

        sbar = b * e - c * d
        tbar = b * d - a * e

        bd = b + d
        ce = c + e

        ad = a + d
        be = b + e

        g = ce - bd
        k = ad - be

        # output
        sout = torch.zeros_like(sbar, device=sbar.device)
        tout = torch.zeros_like(tbar, device=tbar.device)

        # Region classification
        r_conds = torch.stack([(sbar+tbar)<=delta, sbar>=0., tbar>=0., bd>ce, d<0, be>ad])

        # Inside triangle
        r_0 = r_conds[0] & r_conds[1] & r_conds[2] # region 0
        sout[r_0] = sbar[r_0]/delta[r_0]
        tout[r_0] = tbar[r_0]/delta[r_0]

        # region 1
        r_1 = ~r_conds[0] & r_conds[1] & r_conds[2]
        sout[r_1] = torch.clip(g[r_1]/h[r_1], 0., 1.)
        tout[r_1] = 1 - sout[r_1]

        # region 2
        r_2 = ~r_conds[0] & ~r_conds[1] & r_conds[2]
        r_2a = r_2 & r_conds[3] # region 2-a
        sout[r_2a] = torch.clip(g[r_2a] / h[r_2a], 0., 1.)
        tout[r_2a] = 1 - sout[r_2a]
        r_2b = r_2 & ~r_conds[3] # region 2-b
        tout[r_2b] = torch.clip(-e[r_2b]/c[r_2b], 0., 1.) # Note. sout=0 in r_2b

        # region 3
        r_3 = r_conds[0] & ~r_conds[1] & r_conds[2]
        tout[r_3] = torch.clip(-e[r_3] / c[r_3], 0., 1.)  # Note. sout=0 in r_3

        # region 4
        r_4 = r_conds[0] & ~r_conds[1] & ~r_conds[2]
        r_4a = r_4 & r_conds[4] # region 4-a
        sout[r_4a] = torch.clip(-d[r_4a]/a[r_4a], 0., 1.) # Note tout=0 in r_4a
        r_4b = r_4 & ~r_conds[4] # region 4-b
        tout[r_4b] = torch.clip(-e[r_4b] / c[r_4b], 0., 1.)  # Note sout=0 in r_4b

        # region 5
        r_5 = r_conds[0] & r_conds[1] & ~r_conds[2]
        sout[r_5] = torch.clip(-d[r_5]/a[r_5], 0., 1.) # Note tout=0 in r_5

        # region 6
        r_6 = ~r_conds[0] & r_conds[1] & ~r_conds[2]
        r_6a = r_6 & r_conds[5]
        tout[r_6a] = torch.clip(k[r_6a]/h[r_6a], 0., 1.)
        sout[r_6a] = 1 - tout[r_6a]
        r_6b = r_6 & ~r_conds[5]
        tout[r_6b] = torch.clip(-d[r_6b]/a[r_6b], 0., 1.) # Note sout=0 in r_6b

        # Sanity check
        # Should be false all
        # print(r_1, r_2, r_3, r_4, r_5, r_6)
        # print(r_2a, r_2b, r_4a, r_4b, r_6a, r_6b)
        assert not torch.sum(r_1 * r_2 * r_3 * r_4 * r_5 * r_6)
        assert not torch.sum((r_2a * r_2b) + (r_4a * r_4b) + (r_6a * r_6b))

        # assert not r_1 & r_2 & r_3 & r_4 & r_5 & r_6
        # assert not (r_2a & r_2b) | (r_4a & r_4b) | (r_6a & r_6b)

        Pout = B + sout[...,None]*E0 + tout[...,None]*E1

        return Pout, (sout, tout)
