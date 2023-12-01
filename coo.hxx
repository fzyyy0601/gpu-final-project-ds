#include<pair>

typedef vertex_t long long;
typedef edge_t pair<vertex_t,vertex_t> ;
typedef weight_t long long;

class coo{

    vertex_t *v_array;
    edge_t *e_array;
    weight_t *w_array;

public:
    /*
    1
    */
    void init(int vn,int en,int wn){

    }
    /*
    2
    */
    int get_number_of_vertices() const {

        return 0;
    }
    /*
    3
    */
    int get_number_of_edges(){

    }
    /*
    4
    */
    int get_number_of_neighbors(vertex_type x){

    }
    /*
    1
    */
    void set(){

    }
    /*
    2
    */
    vertex_t get_source_vertex(){

    }
    /*
    3
    */
    vertex_t get_destination_vertex(){

    }
    /*
    4
    */
    edge_t get_edge(edge_t){

    }
    /*
    1
    */
    vertex_t get_vertex(vertex_t){

    }
    /*
    2
    */
    edge_t get_weight(edge_t){

    }
    /*
    3
    */
    int get_in_degree(vertex_t){

    }
    /*
    4
    */
    int get_out_degree(vertex_t){

    }
    /*
    1
    */
    edge_t insert_edge(edge_t){

    }
    /*
    2
    */
    vertex_t insert_vertex(edge_t){

    }
    /*
    3
    */
    edge_t delete_edge(edge_t){

    }
    /*
    4
    */
    vertex_t deete_vertex(vertex_t){
        
    }
};