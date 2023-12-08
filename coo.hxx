#include <utility>

template<typename weight_t> class coo{
    /* whether nodes i exits, v_d[i] == 1 means node i is in the graph*/
    bool *v_d;
    /* source of each edge */
    size_t *row_idx_d;
    /* target of each edge */
    size_t *col_idx_d;
    /* wegith of each edge*/
    weight_t *value_d;
    /* number of vertices */
    size_t v_num_d;
    /* number of edges */
    size_t e_num_d;
    /* max number of nodes */
    size_t MAX_d;
    /* deleted slots in the row_idx_d */
    size_t *deleted_d;
    /* head of the deleted_d */
    size_t head;
    /* tail of the deleted_d */
    size_t tail;

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
    size_t get_number_of_vertices() {
        size_t x = get_number_of_vertices_d<<<1, 1>>>(v_num_d);
        return x;
    }
<<<<<<< HEAD

    __device__
    size_t get_number_of_vertices_d (size_t v_num_d)  {
        size_t x = v_num_d;
        return v_num_d;
    }
=======
>>>>>>> 2944830e58723cb7b81abf501cbd6cdadf1ffaf3
    /*
    3
    */
    __host__ __device__
    int get_number_of_edges(){
<<<<<<< HEAD

=======
        return e_size_t;
>>>>>>> 2944830e58723cb7b81abf501cbd6cdadf1ffaf3
    }
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