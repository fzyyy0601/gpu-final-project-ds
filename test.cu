#include "graph.hxx"
#include "coo.hxx"
#include<iostream>

/*
insert nodes: 1 5 7 9 10
insert edge: (1,5) (7,9) (7,10)
vertex=[1,5,7,9,10]
row=[0,2,2]
col=[1,3,4]

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

    /* test initialization */
    printf("Start initialize!\n");
    g.init(v_list_h,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);

    /* test print*/
    g.print();

    /* test get number of vertices and edges */
    printf("\ntest get number of vertices and edges\n");
    printf("number of vertices: %lu, ", g.get_number_of_vertices());
    printf("number of edges: %lu\n", g.get_number_of_edges());

    /* test find vertex and edge */
    printf("\ntest find vertex and edge\n");
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(1));
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(0));
    printf("Is (1, 2) in the graph? %d \n", (int)g.check_edge(1, 2));
    printf("Is (1, 5) in the graph? %d \n", (int)g.check_edge(1, 5));

    /* test insert edge */
    printf("\ntest insert edge\n");
    size_t row_h=1,col_h=2;
    printf("Insert success? %d\n",(int)g.insert_edge(row_h,col_h,3));

    /* test print*/
    g.print();

    row_h=7,col_h=8;
    printf("Insert sueccess? %d\n",(int)g.insert_edge(row_h,col_h,10));

    /* test print*/
    g.print();

    /* test modify configuration */
    printf("\n test modify config \n");
    g.modify_config(8,8);

    /* test print*/
    g.print();
    return 0;
}
