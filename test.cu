#include "graph.hxx"
#include "coo.hxx"
#include<iostream>

/*
*/

int main(){
    graph<int,coo> g;
    size_t v_list_h[]={1,2,3,4,5,6,8};
    size_t v_num_h=7;
    size_t row_idx_h[]={1,1,4,3,5};
    size_t col_idx_h[]={2,3,5,4,6};
    int value_h[]={5,7,2,4,6};
    size_t e_num_h=5;
    size_t number_of_blocks=4;
    size_t threads_per_block=4;
    size_t MAX_h=10;
    printf("\nv_list_h: ");
    for(int i=0;i<v_num_h;i++){
        printf("%lu ",v_list_h[i]);
    }
    printf("\nrow_idx_h: ");
    for(int i=0;i<MAX_h;i++){
        printf("%lu ",row_idx_h[i]);
    }
    printf("\ncol_idx_h: ");
    for(int i=0;i<MAX_h;i++){
        printf("%lu ",col_idx_h[i]);
    }
    printf("\nvalue_h  : ");
    for(int i=0;i<MAX_h;i++){
        printf("%d ",value_h[i]);
    }
    printf("\n");
    g.init(v_list_h,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);
    return 0;
}
