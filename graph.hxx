#include<stdio.h>

template<typename weight_t,
template<typename> typename graph_view_t> class graph {

public:
    graph_view_t<weight_t> graph_t;

    void init(size_t* v_list_h ,size_t v_num_h ,size_t* row_idx_h ,size_t* col_idx_h ,weight_t* value_h ,size_t e_num_h ,size_t number_of_blocks ,size_t threads_per_block ,size_t MAX_h){
        printf("Init\n");
        graph_t.init(v_list_h ,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);
    }

};