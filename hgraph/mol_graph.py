import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from hgraph.chemutils import *
from hgraph.nnutils import *
import numpy as np
from collections import Counter

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 50

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls = self.find_clusters()
        self.mol_tree = self.tree_decomp()
        self.order = self.label_tree()

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: #special case
            return [(0,)], [[0]]

        #separately define proline, because the backbone creates double recognition
        proline = 'O=CC1CCCN1'
        proline_matches = mol.GetSubstructMatches(Chem.MolFromSmiles(proline),useChirality=True)
        
        #find all amino acids
        amino_acid_motif = 'NCC=O'
        matches = mol.GetSubstructMatches(Chem.MolFromSmiles(amino_acid_motif),useChirality=True)
        
        #get matches with overlapping nitrogens for cluster identification (this method will not get the C-terminal amino acid)
        amino_acid_n = 'NCC(=O)N'
        matches_n = mol.GetSubstructMatches(Chem.MolFromSmiles(amino_acid_n),useChirality=True)
        matches_n = [match_n for match_n in matches_n]

        #get matches with overlapping carboynl oxygens for cluster identification (this method will not get the N-terminal amino acid)
        amino_acid_c = 'O=CCNC'
        matches_c = mol.GetSubstructMatches(Chem.MolFromSmiles(amino_acid_c),useChirality=True)

        #find matches that aren't prolines
        if len(proline_matches) == 0:
            matches_without_prolines = matches
            matches_n_no_prolines = matches_n
            matches_c_no_prolines = matches_c
            backbone_atoms = np.hstack(np.unique(np.concatenate((np.hstack(matches_n_no_prolines),np.hstack(matches_c_no_prolines),np.hstack(mol.GetSubstructMatches(Chem.MolFromSmiles('NCC(O)=O'),useChirality=True))))))
        else:
            matches_without_prolines = [match for match in matches if not any([x in np.hstack(proline_matches) for x in match])]
            matches_n_no_prolines = [match for match in matches_n if sum([x in np.hstack(proline_matches) for x in match]) < 2]
            matches_c_no_prolines = [match for match in matches_c if sum([x in np.hstack(proline_matches) for x in match]) < 2]
            backbone_atoms = np.hstack(np.unique(np.concatenate((np.hstack(matches_n_no_prolines),np.hstack(matches_c_no_prolines),np.hstack(proline_matches),np.hstack(mol.GetSubstructMatches(Chem.MolFromSmiles('NCC(O)=O'),useChirality=True))))))

        #split up side chain atoms and backbone atoms
        sidechain_atoms = np.setdiff1d(np.array(range(1,n_atoms)),backbone_atoms)
        
        #identify the c and n terminus by finding where the amino acid backbone begins and ends
        matches_double_overlap = mol.GetSubstructMatches(Chem.MolFromSmiles('NCC(NCC=O)=O'),useChirality=True)
        matches_double_overlap = [m for m in matches_double_overlap if not any([x in sidechain_atoms for x in m])]
        count = Counter(np.hstack(matches_double_overlap))
        matches_double_overlap = [m for m in matches_double_overlap if not all(count[x] > 1 for x in m )]
        if len(matches_double_overlap) == 2:
            n_terminus = [x for x in matches_double_overlap[0] if count[x] == 1]
            c_terminus = [x for x in matches_double_overlap[1] if count[x] == 1]
        else:
            print('could not identify termini correctly')
             
        #identify the termini clusters
        #nterm cluster
        if len(proline_matches) != 0 and any([x in np.hstack(proline_matches) for x in n_terminus]): #n_terminus is a proline
            n_terminus = [np.array(proline) for proline in proline_matches if any([x in n_terminus for x in proline])][0]
        else: #n_terminus is not a proline
            n_terminus = [np.array(match) for match in matches if any([x in n_terminus for x in match])][0]

        #cterm cluster
        if len(proline_matches) != 0 and any([x in np.hstack(proline_matches) for x in c_terminus]):
            c_terminus = np.array(mol.GetSubstructMatches(Chem.MolFromSmiles('CN1[C@H](C(O)=O)CCC1'),useChirality=True))[0]
        else: #c terminus is not a proline
            c_terminus = np.array(mol.GetSubstructMatches(Chem.MolFromSmiles('CNCC(=O)O'),useChirality=True))[0]

        #non-terminus prolines
        prolines_non_terminal = [proline for proline in proline_matches if sum([(x in c_terminus) or (x in n_terminus) for x in proline]) < 3]
        if len(prolines_non_terminal) > 0:
            prolines_not_terminal_overlap = mol.GetSubstructMatches(Chem.MolFromSmiles('O=CC1CCCN1C'),useChirality=True)
            prolines_not_terminal_overlap = [proline for proline in prolines_not_terminal_overlap if any([x in np.hstack(prolines_non_terminal) for x in proline])]
        else:
            prolines_not_terminal_overlap = []
        
        #define clusters from backbones, the termini, and any prolines
        clusters = list()
        [clusters.append(match) for match in matches_c_no_prolines if sum([x in c_terminus for x in match]) < 2]
        clusters.append(tuple(c_terminus))
        clusters.append(tuple(n_terminus))
        [clusters.append(proline) for proline in prolines_not_terminal_overlap]
        
        #define alpha carbons and sets of backbone/ sideatoms without and with them
        c_alphas = np.array(matches)[:,1]
        c_alphas_neighbors = [mol.GetAtomWithIdx(int(x)).GetNeighbors() for x in c_alphas]
        sidechain_atoms_with_calphas = np.union1d(sidechain_atoms,c_alphas)
        backbone_atoms_without_calphas = np.setdiff1d(backbone_atoms,c_alphas)
        
        ##recursively identify alpha carbons, and map the atom indices that contain their side chains
        for index,c_alpha in enumerate(c_alphas): #iterate throguh each alpha carbon
            c_alpha_neighbors = mol.GetAtomWithIdx(int(c_alpha)).GetNeighbors() #get the neighbors of given c_alpha. serves as starting point for molecular search
            if len(proline_matches) != 0 :
                if c_alpha in np.hstack(proline_matches):
                    continue
            for neighbor in c_alpha_neighbors: #for each atom next to a c_alpha, traverse it if it's not a backbone atom and it's also a carbon
                if (neighbor.GetIdx() not in backbone_atoms) and (neighbor.GetAtomicNum() == 6):
                    side_chain_atoms = np.array([neighbor.GetIdx()]) #define the side chain of this c_alpha as a set, starting over at each new c_alpha
                    new_neighbors = neighbor.GetNeighbors() # atoms to explore
                    atoms_to_examine = np.array([]) #next atoms to explore
                    while(len(new_neighbors) > 0): #if there are new neighbors from the previous step, proceed with side chain traversal
                        for new_neighbor in new_neighbors:
                            if (new_neighbor.GetIdx() in c_alphas) and (new_neighbor.GetIdx() not in sidechain_atoms): #if the next atom is a c_alpha, include it (this works for non-stapled and stapled residues)
                                side_chain_atoms = np.append(side_chain_atoms,new_neighbor.GetIdx())
                            elif (new_neighbor.GetIdx() not in side_chain_atoms) and (new_neighbor.GetIdx() not in backbone_atoms): #if it's a regular side chain, add it, and all the new neighbors
                                side_chain_atoms = np.append(side_chain_atoms,new_neighbor.GetIdx())
                                atoms_to_examine = np.append(atoms_to_examine,new_neighbor.GetIdx())
                            elif new_neighbor.GetIdx()  in side_chain_atoms: #skip previously explored side chain atoms
                                continue
                            elif new_neighbor.GetIdx() in backbone_atoms: #skip previously explored backbone atoms
                                continue
                        if len(atoms_to_examine) == 0: #if there's no more atoms to examine, end traversal
                            break
                        else:
                            new_neighbors = np.hstack([mol.GetAtomWithIdx(int(atom)).GetNeighbors() for atom in atoms_to_examine]) #define the next step of neighbors as the atoms to examine
                            atoms_to_examine = np.array([])
                    if sum([x in np.hstack(clusters) for x in side_chain_atoms]) < 3: #if most of this side chain has already been added as a cluster, do not add it (this commonly happens when a stapled side chain is traversed from both sides)
                        clusters.append(tuple([x for x in side_chain_atoms]))

        #convert cluster atoms to int
        clusters = [tuple([int(atom) for atom in cluster]) for cluster in clusters]

        #for each atom in the molecule, find which clusters it's a part of
        atom_cls = [[] for i in range(n_atoms)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)
        if len(set(np.hstack(clusters))) != len(mol.GetAtoms()): #every atom must be part of a cluster
            print(mol)
            assert len(set(np.hstack(clusters))) == len(mol.GetAtoms())
        return clusters, atom_cls

    def tree_decomp(self):
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) )
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            bonds = [c for c in nei_cls if len(clusters[c]) == 2]
            rings = [c for c in nei_cls if len(clusters[c]) > 4] #need to change to 2

            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters.append([atom])
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                clusters.append([atom]) #temporary value, need to change
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)
            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters[c1]) & set(clusters[c2])
                        graph.add_edge(c1, c2, weight = len(inter))

        n, m = len(graph.nodes), len(graph.edges)
        # if n - m <= 1:
        #     print(self.smiles)
        assert n - m <= 1 #must be connected
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph)

    def label_tree(self):
        #dfs method from mol_graph original
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            for idx,y in enumerate(sorted_child):
                if ((x,y,1) in order) or ((y,x,1) in order):
                    # print('trying to add duplicate path to order')
                    continue
                else:
                    self.mol_tree[x][y]['label'] = 0 
                    self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                    prev_sib[y] = sorted_child[:idx] 
                    prev_sib[y] += [x, fa] if fa >= 0 else [x]
                    order.append( (x,y,1) )
                    dfs(order, pa, prev_sib, y, x)
                    # print((y,x,0))
                    order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) ) #last backtrack at root
        
        mol = get_mol(self.smiles)
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1 )

        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            cls = [int(cl) for cl in cls]
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            inter_atoms = set([int(inter_atom) for inter_atom in inter_atoms])
            cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [int(a) for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order
       
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab):
        mol_batch = [MolGraph(x) for x in mol_batch]
        tree_tensors, tree_batchG = MolGraph.tensorize_graph([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)

        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

if __name__ == "__main__":
    import sys
    
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1','O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1', 'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1', 'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br', 'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F', 'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1', 'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1', 'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in sys.stdin:#test_smiles:
        print(s.strip("\r\n "))
        #mol = Chem.MolFromSmiles(s)
        #for a in mol.GetAtoms():
        #    a.SetAtomMapNum( a.GetIdx() )
        #print(Chem.MolToSmiles(mol))

        hmol = MolGraph(s)
        print(hmol.clusters)
        #print(list(hmol.mol_tree.edges))
        print(nx.get_node_attributes(hmol.mol_tree, 'label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'inter_label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'assm_cands'))
        #print(hmol.order)
