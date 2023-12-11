#include "graph.hxx"
#include "coo_d.hxx"
#include <iostream>
#include <vector>

/* 
This is the graph we test
    insert nodes: 1 5 7 9 10
    insert edge: (1,5) (7,9) (7,10)
    vertex=[1,5,7,9,10]
    row=[0,2,2]
    col=[1,3,4]
*/

int main(){
    /* intital graph configuration */
    graph<int,coo_d> g;           // create a graph in device
    size_t v_list_h[]={1,2,3,4,5,6,8};  // intinal vertex list
    size_t v_num_h=7;           // initial number of vertices
    size_t row_idx_h[]={1,1,4,3,5}; // initial row index (source of each edge)
    size_t col_idx_h[]={2,3,5,4,6}; // initial col index (target of each edge)
    int value_h[]={5,7,2,4,6};  // initial weight for each edge
    size_t e_num_h=5;           // intinal number of edges
    size_t MAX_h=10;            // max number of edges
    
    /* initial configuration for the kernel call */
    size_t number_of_blocks=4;  
    size_t threads_per_block=4;

    /* print graph config for initialization*/
    printf("Graph config for initialization:");
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

    printf("----------------------------Test begin------------------------------------\n");

    /* test initialization */
    printf("Start initialize!\n");
    g.init(v_list_h,v_num_h,row_idx_h,col_idx_h,value_h,e_num_h,number_of_blocks,threads_per_block,MAX_h);

    /* test print*/
    printf("\nTEST - print graph:\n");
    g.print();

    /* test insert edge */
    printf("\nTEST - insert edge: \n");
    size_t row_h=1,col_h=2;
    printf("Insert (%lu -> %lu, 3), success? %d\n", row_h, col_h, (int)g.insert_edge(row_h,col_h,3));
    g.print();
    row_h=6,col_h=8;
    printf("Insert (%lu -> %lu, 10), success? %d\n", row_h, col_h,(int)g.insert_edge(row_h,col_h,10));
    g.print();

    /* test check edge */
    printf("\nTEST - check edge: \n");
    printf("Is (7, 8) in the graph? %d \n", (int)g.check_edge(7, 8));

    /* test modify configuration */
    printf("\nTEST - modify config to (8, 8) \n");
    g.modify_config(8,8);
    g.print_config();

    /* test insert edge */
    printf("\nTEST - delete edge\n");
    row_h=7,col_h=8;
    printf("Delete (%lu -> %lu), success? %d\n", row_h, col_h,(int)g.delete_edge(row_h,col_h));
    g.print();
    row_h=6,col_h=8;
    printf("Delete (%lu -> %lu), success? %d\n", row_h, col_h,(int)g.delete_edge(row_h,col_h));
    g.print();

    // printf("----------------------------Test 1 end--------------------------------------\n");

    // printf("----------------------------Test 2 begin------------------------------------\n");
    /* 2 test get number of vertices and edges */
    printf("\nTEST - get number of vertices and edges\n");
    printf("number of vertices: %lu, ", g.get_number_of_vertices());
    printf("number of edges: %lu\n", g.get_number_of_edges());

    /* 2 test find vertex and edge */
    printf("\nTEST - find vertex and edge\n");
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(1));
    printf("Is 1 in the graph? %d \n", (int)g.check_vertex(0));
    printf("Is (1, 2) in the graph? %d \n", (int)g.check_edge(1, 2));
    printf("Is (1, 5) in the graph? %d \n", (int)g.check_edge(1, 5));

    /* 2 test find vertex and edge */
    printf("\nTEST - get_weight. If not found, return -1\n");
    printf("the value of (1, 2) is %d \n", g.get_weight(1, 2, -1));
    printf("the value of (1, 5) is %d \n", g.get_weight(1, 5, -1));

    /* 2 insert vertex */
    printf("\nTEST - insert vertex\n");
    printf("insert 0? %d\n", (int)g.insert_vertex(0));
    printf("insert 1? %d\n", (int)g.insert_vertex(1));
    g.print();
    // printf("----------------------------Test 2 end-------------------------------------\n");

    // printf("----------------------------Test 3 begin------------------------------------\n");
    /* 3 test get number of in degrees */
    printf("\nTEST - get in degrees\n");
    printf("number of in-degrees for vertex 3: %lu, \n", g.get_in_degree(3));

    /* 3 test get number of out degrees */
    printf("\nTEST - get out degrees\n");
    printf("number of out-degrees for vertex 1: %lu, \n", g.get_out_degree(1));

    /* 3 test get number of neighbors */
    printf("\nTEST - get number of neighbors\n");
    printf("number of neighbors for vertex 6: %lu, \n", g.get_num_neighbors(6));

    // printf("----------------------------Test 3 end-------------------------------------\n");

    // printf("----------------------------Test 4 begin------------------------------------\n");
    printf("\nTEST - delete vertex\n");
    size_t v_del = 3;
    printf("Delete vertex %lu, success? %d\n", v_del, (int)g.delete_vertex(v_del));
    g.print();

    // printf("----------------------------Test 4 end-------------------------------------\n");
    // printf("----------------------------Test 5 begin------------------------------------\n");
    printf("\nTEST - get destination of vertex\n");

    printf("get destination of 1:\n");
    std::vector<size_t> destination = g.get_destination_vertex(1);
    if(destination.empty()) 
        printf("No destination vertex for 1.\n");
    else{
        printf("destination of vertex 1: ");
        for(int i=0; i < destination.size(); i++) 
            printf("%lu\t",destination[i]);
        printf("\n");
    }

    printf("\nTEST - get source of vertex\n");
    printf("get source of 1:\n");
    std::vector<size_t> source = g.get_source_vertex(1);
    if(source.empty()) 
        printf("No source vertex for 1.\n");
    else{
        for(int i=0; i < source.size(); i++) 
            printf("%lu\t",source[i]);
	}
    printf("get source of 6:\n");
    std::vector<size_t> source1 = g.get_source_vertex(6);
    if(source1.empty()) 
        printf("No source vertex for 6.\n");
    else{
        printf("sources of vertex 6: ");
        for(int i=0; i < source1.size(); i++) 
            printf("%lu\t",source1[i]);
        printf("\n");
	}
    g.print();

    printf("----------------------------Test end-------------------------------------\n");

    return 0;
}
