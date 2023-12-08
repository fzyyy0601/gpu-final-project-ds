#include <utility>

template<typename weight_t> class coo{
    size_t *v_list_d;
    size_t *row_idx_d;
    size_t *col_idx_d;
    weight_t *value_d;
    size_t v_size_d;
    size_t e_size_d;
    size_t MAX_d;
    size_t *deleted_d;
    
public:
    /*s
    1
    */
    __host__ 
    void init(size_t *v_list, std::tuple<size_t,size_t,weight_t> *e_list, size_t MAX){

    }

    __device__
    void init_d(){}

    /*
    2
    */
    __host__ __device__
    size_t get_number_of_vertices() const {
        return v_size_t;
    }
//     /*
//     3
//     */
//     __host__ __device__
//     int get_number_of_edges(){

//     }
//     /*
//     4
//     */
//     __host__ __device__
//     int get_number_of_neighbors(vertex_type x){

//     }
    // /*
    // 1
    // */
    // void set(){

    // }
    // /*
    // 2
    // */
    // vertex_t get_source_vertex(){

    // }
    // /*
    // 3
    // */
    // vertex_t get_destination_vertex(){

    // }
    // /*
    // 4
    // */
    // edge_t get_edge(edge_t){

    // }
    // /*
    // 1
    // */
    // vertex_t get_vertex(vertex_t){

    // }
    // /*
    // 2
    // */
    // edge_t get_weight(edge_t){

    // }
    // /*
    // 3
    // */
    // int get_in_degree(vertex_t){

    // }
    // /*
    // 4
    // */
    // int get_out_degree(vertex_t){

    // }
    // /*
    // 1
    // */
    // edge_t insert_edge(edge_t){

    // }
    // /*
    // 2
    // */
    // vertex_t insert_vertex(edge_t ){

    // }
    // /*
    // 3
    // */
    // edge_t delete_edge(edge_t){

    // }
    // /*
    // 4
    // */
    // vertex_t delete_vertex(vertex_t){
        
    // }
};