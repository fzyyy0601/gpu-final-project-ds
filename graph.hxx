#include<stdio.h>

template<typename weight_t,
template<typename> typename graph_view_t> class graph {

public:
    graph_view_t<weight_t> graph_t;

    void init(size_t* v_list_h ,size_t v_num_h ,size_t* row_idx_h ,size_t* col_idx_h ,weight_t* value_h ,size_t e_num_h ,size_t number_of_blocks ,size_t threads_per_block ,size_t MAX_h){
        printf("Init\n");
        graph_t.init(v_list_h ,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);
    }

    __host__ __device__
    size_t get_number_of_vertices() const {
        return graph_t.get_number_of_vertices();
    }

    __host__ __device__
    int get_num_neighbors(vertex_type x){
        return graph_t.get_num_neighbors(x);
    }

//     __host__ __device__
//     edge_t get_number_of_edges(){
//         return graph_t.get_number_of_edges();
//     }

//     vertex_t get_source_vertex(){

//     }

//     vertex_t get_destination_vertex(){

//     }

//     edge_t get_edge(edge_t){

//     }

//     vertex_t get_vertex(vertex_t){

//     }

//     edge_t get_weight(edge_t){

//     }
    __host__ __device__
    int get_in_degree(vertex_t v){
        return graph_t.get_in_degree(v);
    }

    __host__ __device__
    int get_out_degree(vertex_t v){
        return graph_t.get_out_degree(v);
    }

//     edge_t insert_edge(edge_t){

//     }

//     vertex_t insert_vertex(edge_t){

//     }

//     edge_t delete_edge(edge_t){

//     }

//     vertex_t delete_vertex(vertex_t){
        
//     }

};