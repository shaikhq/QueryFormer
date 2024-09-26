import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *
import logging
import ipdb

class PlanTreeDataset(Dataset):
    def __init__(self, json_df : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):
        #ipdb.set_trace()

        logging.info('Initializing PlanTreeDataset')

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
            
        # length is the number of queries in the training set
        self.length = len(json_df)
        logging.info('self.length = len(json_df): {}'.format(self.length))

        # train = train.loc[json_df['id']]
        # TODO: At the top level dictionary, add Execution Time - DONE
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        logging.info('nodes.type: {}'.format(type(nodes)))
        logging.info('number of nodes: {}'.format(len(nodes)))
        logging.info('type of the first element in the list nodes: {}'.format(type(nodes[0])))
        logging.info('keys in the first dictionary in the nodes list: {}'.format(nodes[0].keys()))

        self.cards = [node['Actual Rows'] for node in nodes]
        logging.info('self.cards: {}'.format(self.cards))
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
            
        # TODO: check if I need to change the min and max values in both normalizers
        # TODO: for now, I am not normalizing it since I have only 1 training query
        # Original line
        # self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        # temporary line:
        self.card_labels = torch.from_numpy(np.array(self.cards))

        logging.info('self.card_labels: {}'.format(self.card_labels))

        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
            
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both': ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')
        
        #TODO: change this to get the query ids - should be done, but needs verification
        idxs = list(json_df['id'])
        logging.info('idxs: {}'.format(idxs))
            
        self.treeNodes = [] ## for mem collection

        self.collated_dicts = [self.js_node2dict(i,node) for i,node in zip(idxs, nodes)]
        
        # TODO: db2 dataset prep - uncomment below
        logging.info('collated_dicts: {}'.format(type(self.collated_dicts)))
        logging.info('collated_dicts (list) len: {}'.format(len(self.collated_dicts)))
        # logging.info('collated_dicts (list) item type: {}'.format(type(self.collated_dicts[0])))
        logging.info('type(self.collated_dicts[0]): {}'.format(type(self.collated_dicts[0])))
        logging.info('self.collated_dicts[0].keys(): {}'.format(self.collated_dicts[0].keys()))
        logging.info("self.collated_dicts[0].keys()['x']: {}".format(self.collated_dicts[0]['x']))
        logging.info("self.collated_dicts[0].keys()['x'].shape: {}".format(self.collated_dicts[0]['x'].shape))
        logging.info("self.collated_dicts[0].keys()['attn_bias']: {}".format(self.collated_dicts[0]['attn_bias']))
        logging.info("self.collated_dicts[0].keys()['attn_bias'].shape: {}".format(self.collated_dicts[0]['attn_bias'].shape))
        logging.info("self.collated_dicts[0].keys()['rel_pos']: {}".format(self.collated_dicts[0]['rel_pos']))
        logging.info("self.collated_dicts[0].keys()['rel_pos'].shape: {}".format(self.collated_dicts[0]['rel_pos'].shape))
        logging.info("self.collated_dicts[0].keys()['heights']: {}".format(self.collated_dicts[0]['heights']))
        logging.info("self.collated_dicts[0].keys()['heights'].shape: {}".format(self.collated_dicts[0]['heights'].shape))
        logging.info("labels: {}".format(self.labels))
        #TODO: db2 dataset prep - uncomment above

        logging.info('printing the tensors of the 2nd training query')
        logging.info('collated_dicts: {}'.format(type(self.collated_dicts)))
        logging.info('collated_dicts (list) len: {}'.format(len(self.collated_dicts)))
        # logging.info('collated_dicts (list) item type: {}'.format(type(self.collated_dicts[0])))
        # logging.info('type(self.collated_dicts[0]): {}'.format(type(self.collated_dicts[1])))
        # logging.info('self.collated_dicts[0].keys(): {}'.format(self.collated_dicts[1].keys()))
        # logging.info("self.collated_dicts[0].keys()['x']: {}".format(self.collated_dicts[1]['x']))
        # logging.info("self.collated_dicts[0].keys()['x'].shape: {}".format(self.collated_dicts[1]['x'].shape))
        # logging.info("self.collated_dicts[0].keys()['attn_bias']: {}".format(self.collated_dicts[1]['attn_bias']))
        # logging.info("self.collated_dicts[0].keys()['attn_bias'].shape: {}".format(self.collated_dicts[1]['attn_bias'].shape))
        # logging.info("self.collated_dicts[0].keys()['rel_pos']: {}".format(self.collated_dicts[1]['rel_pos']))
        # logging.info("self.collated_dicts[0].keys()['rel_pos'].shape: {}".format(self.collated_dicts[1]['rel_pos'].shape))
        # logging.info("self.collated_dicts[0].keys()['heights']: {}".format(self.collated_dicts[1]['heights']))
        # logging.info("self.collated_dicts[0].keys()['heights'].shape: {}".format(self.collated_dicts[1]['heights'].shape))

        logging.info('length of treeNodes: {}'.format(len(self.treeNodes)))


        logging.info('PlanTreeDataset initialized')

    def js_node2dict(self, idx, node):
        logging.info('beginning js_node2dict(self, idx, node): returns a dictionary of 4 tensors per query plan')
        logging.info("returns a collated_dict of 4 tensors: 'x', 'attn_bias', 'rel_pos', 'heights ")

        treeNode = self.traversePlan(node, idx, self.encoding)

        
        _dict = self.node2dict(treeNode)
        logging.info("_dict's features: {}".format(_dict['features']))
        logging.info("_dict's features shape: {}".format(_dict['features'].shape))
        
        #TODO: js_node2dict - uncomment below

        logging.info("_dict's heights: {}".format(_dict['heights']))
        logging.info("_dict's heights shape: {}".format(_dict['heights'].shape))

        logging.info("_dict's adjacency_list: {}".format(_dict['adjacency_list']))
        logging.info("_dict's adjacency_list shape: {}".format(_dict['adjacency_list'].shape))

        collated_dict = self.pre_collate(_dict)
        logging.info("collated_dict's features shape: {}".format(collated_dict['x'].shape))
        logging.info("collated_dict's attn_bias shape: {}".format(collated_dict['attn_bias'].shape))
        logging.info("collated_dict's rel_pos shape: {}".format(collated_dict['rel_pos'].shape))
        logging.info("collated_dict's heights shape: {}".format(collated_dict['heights'].shape))
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict
        #TODO: js_node2dict - uncomment above
        return {}


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)

        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    # traversing the plan using BFS - Breadth-First Search. 
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        #logging.info('nodeType: {}'.format(nodeType))

        typeId = encoding.encode_type(nodeType)
        #logging.info('typeId: {}'.format(typeId))

        card = None #plan['Actual Rows']
        #TODO: adapt formatFilter to work with a list of filters
        filters, alias = formatFilter(plan)
        logging.info('formatFilter - filters: {}'.format(filters))
        logging.info('formatFilter - alias: {}'.format(alias))

        filters_encoded = encoding.encode_filters(filters, alias)

        #TODO: check if I need to add any key for the JOIN nodes
        join = formatJoin(plan)
        logging.info('formatJoin - join: {}'.format(join))
        joinId = encoding.encode_join(join)
        logging.info('formatJoin - joinId: {}'.format(joinId))

        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        logging.info('printing node features')
        logging.info('node.typeId: {}'.format(root.typeId))
        logging.info('node.join: {}'.format(root.join))
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 



def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3,3-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 

    mask = np.zeros(3)
    mask[:num_filter] = 1

    type_join = np.array([node.typeId, node.join])
    
    #TODO: Uncomment the following 3 lines of code, which encode filter values using hist
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)
    logging.info('hists = filterDict2Hist(hist_file, node.filterDict, encoding): printing hists {}'.format(hists))
    logging.info('len(hists) {}'.format(len(hists)))

    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    logging.info('node.table: {}'.format(node.table))
    

    if node.table_id == 0:
        sample = np.zeros(1000)
    # TODO: I temporarily added the following 2 lines of code to crate a fake sample bitmap for TPCDS dataset. I need to remove them after creating 
    # actual sample bitmap
    elif node.table not in table_sample[node.query_id]:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]
    
    #return np.concatenate((type_join,filts,mask))
    
    #TODO: Uncomment below original line
    # return np.concatenate((type_join, filts, mask, hists, table, sample))
    
    logging.info('printing node elements from node2feature method:')
    logging.info('node.type: {}'.format(node.typeId))
    logging.info('node.join: {}'.format(node.join))
    #logging.info('node.filts: {}'.format(node.filts))
    logging.info('node.mask: {}'.format(mask))
    logging.info('node.table: {}'.format(node.table))
    
    if node.table_id > 0:
        logging.info('node.sample: {}'.format(sample))
    
    return np.concatenate((type_join, filts, mask, table, sample))

