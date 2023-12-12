#include <stdio.h>
#include <vector>

template<typename weight_t,
template<typename> typename graph_view_t> class graph {

public:
    graph_view_t<weight_t> graph_t;

    /* 1 initialize the graph */
    void init(size_t* v_list,
                size_t v_num,
                size_t* row_idx,
                size_t* col_idx,
                weight_t* value,
                size_t e_num,
                size_t number_of_blocks,
                size_t threads_per_block,
                size_t MAX){
        graph_t.init(v_list ,v_num,row_idx,col_idx,value,e_num,number_of_blocks,threads_per_block,MAX);
    }

    /* 1 print the graph */
    void print(){
        graph_t.print();
    }

    /* 1 print the graph configuration */
    void print_config(){
        graph_t.print_config();
    }

    /* 1 modify grid size and block size*/
    void modify_config(size_t number_of_blocks,
                        size_t threads_per_block){
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
    bool check_edge(size_t row, 
                    size_t col){
        return graph_t.check_edge(row, col);
    }

    /* 2 if edge is in the graph, return value. Otherwise, return not_found */
    weight_t get_weight(size_t row, 
                        size_t col, 
                        weight_t not_found){
        return graph_t.get_weight(row, col, not_found);
    }

    /* 4 Get neighbors */
    size_t get_num_neighbors(size_t vertex){
        return graph_t.get_num_neighbors(vertex);
    }

    /* 4 Get in degree */
    size_t get_in_degree(size_t vertex){
        return graph_t.get_in_degree(vertex);
    }

    /* 4 Get out degree */
    size_t get_out_degree(size_t vertex){
        return graph_t.get_out_degree(vertex);
    }

    /* 2 insert vertex */
    bool insert_vertex(size_t vertex){
        return graph_t.insert_vertex(vertex);
    }


    /* 1 if edge is in the graph, return False. Otherwise, return True then inseert*/
    bool insert_edge(size_t row,
                    size_t col,
                    weight_t value){
        return graph_t.insert_edge(row,col,value);
    }

    /* 3 Delete edge (row_h,col_h,value_h) */
    bool delete_edge(size_t row,
                        size_t col){
        return graph_t.delete_edge(row,col);
    }

    /* 4 Delete vertex (v_del) */
    bool delete_vertex(size_t v_del){
        return graph_t.delete_vertex(v_del);
    }
    
    /* 3 return list of destination vertex */
    std::vector<size_t> get_destination_vertex(size_t vertex){
        return graph_t.get_destination_vertex(vertex);
    }
    
    /* 3 return list of source vertex */
    std::vector<size_t> get_source_vertex(size_t vertex){
        return graph_t.get_source_vertex(vertex);
    }
};
