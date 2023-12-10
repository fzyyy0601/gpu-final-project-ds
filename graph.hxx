#include<stdio.h>

template<typename weight_t,
template<typename> typename graph_view_t> class graph {

public:
    graph_view_t<weight_t> graph_t;

    /* 1 initialize the graph */
    void init(size_t* v_list_h ,size_t v_num_h ,size_t* row_idx_h ,size_t* col_idx_h ,weight_t* value_h ,size_t e_num_h ,size_t number_of_blocks ,size_t threads_per_block ,size_t MAX_h){
        graph_t.init(v_list_h ,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);
    }

    /* 1 print the graph */
    void print(){
        graph_t.print();
    }

    /* 1 modify grid size and block size*/
    void modify_config(size_t number_of_blocks,size_t threads_per_block){
        graph_t.modify_config(number_of_blocks,threads_per_block);
    }

    /* 2 return the vertex number in graph */
    size_t get_number_of_vertices(){
        return graph_t.get_number_of_vertices();
    }

    /* 2 return the edge number in graph */
    size_t get_number_of_edges(){
        return graph_t.get_number_of_edges();
    }

    /* 2 if vertex is in the graph, return True. Otherwise, return False */
    bool check_vertex(size_t vertex){
        return graph_t.check_vertex(vertex);
    }

    /* 2 if edge is in the graph, return True. Otherwise, return False */
    bool check_edge(size_t row, size_t col){
        return graph_t.check_edge(row, col);
    }

    /* 2 if edge is in the graph, return value. Otherwise, return not_found */
    weight_t get_weight(size_t row, size_t col, weight_t not_found){
        return graph_t.get_weight(row, col, not_found);
    }

    /* 1 if edge is in the graph, return False. Otherwise, return True then inseert*/
    bool insert_edge(size_t row_h,
                    size_t col_h,
                    weight_t value_h){
        return graph_t.insert_edge(row_h,col_h,value_h);
    }

    __host__ __device__
    size_t get_num_neighbors(size_t x){
        return graph_t.get_num_neighbors(x);
    }

    __host__ __device__
    size_t get_in_degree(size_t v){
        return graph_t.get_in_degree(v);
    }

    __host__ __device__
    size_t get_out_degree(size_t v){
        return graph_t.get_out_degree(v);
    }
    // __host__ __device__ 
    vector<size_t> get_destination_vertex(size_t x){
        return graph_t.get_destination_vertex(x);
    }
    // __host__ __device__ 
    vector<size_t> get_source_vertex(size_t x){
        return graph_t.get_source_vertex(x);
    }
};