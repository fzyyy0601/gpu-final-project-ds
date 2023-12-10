#include "graph.hxx"
#include "coo.hxx"
#include<iostream>
#include<vector>

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

    printf("----------------------------Test 1 begin------------------------------------\n");

    /* test initialization */
    printf("Start initialize!\n");
    g.init(v_list_h,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);

    /* test print*/
    g.print();

    /* test insert edge */
    printf("\ntest insert edge\n");
    size_t row_h=1,col_h=2;
    printf("Insert success? %d\n",(int)g.insert_edge(row_h,col_h,3));

    /* test print*/
    g.print();

    printf("Is (7, 8) in the graph? %d \n", (int)g.check_edge(7, 8));

    row_h=6,col_h=8;
    printf("Insert sueccess? %d\n",(int)g.insert_edge(row_h,col_h,10));

    /* test print*/
    g.print();

    /* test modify configuration */
    printf("\ntest modify config \n");
    g.modify_config(8,8);

    /* test insert edge */
    printf("\ntest delete edge\n");

    row_h=7,col_h=8;
    printf("Delete sueccess? %d\n",(int)g.delete_edge(row_h,col_h));

    /* test print*/
    g.print();

    row_h=6,col_h=8;
    printf("Delete sueccess? %d\n",(int)g.delete_edge(row_h,col_h));

    /* test print*/
    g.print();

    printf("----------------------------Test 1 end--------------------------------------\n");

    printf("----------------------------Test 2 begin------------------------------------\n");
    /* 2 test get number of vertices and edges */
    printf("\ntest get number of vertices and edges\n");
    printf("number of vertices: %lu, ", g.get_number_of_vertices());
    printf("number of edges: %lu\n", g.get_number_of_edges());

    /* 2 test find vertex and edge */
    printf("\ntest find vertex and edge\n");
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(1));
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(0));
    printf("Is (1, 2) in the graph? %d \n", (int)g.check_edge(1, 2));
    printf("Is (1, 5) in the graph? %d \n", (int)g.check_edge(1, 5));

    /* 2 test find vertex and edge */
    printf("\ntest get_weight\n");
    printf("the value of (1, 2) is %d \n", g.get_weight(1, 2, -1));
    printf("the value of (1, 5) is %d \n", g.get_weight(1, 5, -1));

    /* 2 insert vertex */
    printf("\n test insert vertex\n");
    printf("insert 0? %d\n", (int)g.insert_vertex(0));
    printf("insert 1? %d\n", (int)g.insert_vertex(1));
    printf("----------------------------Test 2 end-------------------------------------\n");

    printf("----------------------------Test 3 begin------------------------------------\n");
    /* 3 test get number of in degrees */
    printf("\ntest get in degrees\n");
    printf("number of in-degrees for vertex 3: %lu, \n", g.get_in_degree(3));

    /* 3 test get number of out degrees */
    printf("\ntest get out degrees\n");
    printf("number of out-degrees for vertex 1: %lu, \n", g.get_out_degree(1));

    /* 3 test get number of neighbors */
    printf("\ntest get number of neighbors\n");
    printf("number of neighbors for vertex 6: %lu, \n", g.get_num_neighbors(6));

    printf("----------------------------Test 3 end-------------------------------------\n");
    
    printf("----------------------------Test 4 begin-------------------------------------\n");
    printf("\nget source of 1:\n");
    std::vector<size_t> source = g.get_source_vertex(1);
    printf(source.size);
    // for(size_t each: source){
    //     printf("\t%lu", each);
    // }
    // for(auto t: source) cout<<t<<" ";
    printf("\nget destination of 1:\n");
    std::vector<size_t> destination = g.get_destination_vertex(1);
    // for(size_t each: destination){
    //     printf("\t%lu", each);
    // }
    printf("----------------------------Test 4 end-------------------------------------\n");
     

    return 0;
}
