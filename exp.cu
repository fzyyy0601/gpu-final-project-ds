#include "graph.hxx"
#include "coo.hxx"
#include <iostream>
#include <vector>
#include <time.h>

int main(){
    /* set the random seed and clock */
    srand(time(0));
    double time_taken;
    clock_t start, end;
    clock_t T_start, T_end;

    /* initialize an empty graph*/
    graph<int,coo> g;
    size_t v_list[] = {};
    size_t v_num = 0;
    size_t row_idx[] = {};
    size_t col_idx[] = {};
    int value[] = {};
    size_t e_num = 0;
    size_t number_of_blocks = 50;
    size_t threads_per_block = 256;
    size_t MAX = 10000;

    /* print the initial graph we input */
    printf("The initial graph we input: \n");
    printf("v_list: ");
    for(int i = 0; i < v_num; i++){
        printf("%lu ", v_list[i]);
    }
    printf("\n");
    
    printf("row_idx: ");
    for(int i = 0; i < e_num; i++){
        printf("%lu ", row_idx[i]);
    }
    printf("\n");
    
    printf("col_idx: ");
    for(int i = 0; i < e_num; i++){
        printf("%lu ", col_idx[i]);
    }
    printf("\n");
    
    printf("value: ");
    for(int i = 0; i < e_num; i++){
        printf("%d ", value[i]);
    }
    printf("\n");

    printf("\n ----------------------------Test begin------------------------------------\n");

    /* test initialization */
    T_start = clock();
    start = clock();
    g.init(v_list, v_num, row_idx, col_idx, value, e_num, number_of_blocks, threads_per_block, MAX);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time for initialization: %lf s\n", time_taken);

    /* insert 10000 edges */
    start = clock();
    for (int i = 0; i < 10000; i ++) {
        g.insert_vertex(rand() % MAX);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time for inserting 10000 vertices: %lf s\n", time_taken);

    /* insert 10000 edges */
    start = clock();
    for (int i = 0; i < 10000; i ++) {
        g.insert_edge(rand() % MAX, rand() % MAX, rand() % MAX);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time for inserting 10000 edges: %lf s\n", time_taken);


    // /* test print*/
    // g.print();

    // printf("Is (7, 8) in the graph? %d \n", (int)g.check_edge(7, 8));

    // row_h=6,col_h=8;
    // printf("Insert sueccess? %d\n",(int)g.insert_edge(row_h,col_h,10));

    // /* test print*/
    // g.print();

    // /* test modify configuration */
    // printf("\ntest modify config \n");
    // g.modify_config(8,8);

    // /* test insert edge */
    // printf("\ntest delete edge\n");

    // row_h=7,col_h=8;
    // printf("Delete sueccess? %d\n",(int)g.delete_edge(row_h,col_h));

    // /* test print*/
    // g.print();

    // row_h=6,col_h=8;
    // printf("Delete sueccess? %d\n",(int)g.delete_edge(row_h,col_h));

    // /* test print*/
    // g.print();

    // printf("----------------------------Test 1 end--------------------------------------\n");

    // printf("----------------------------Test 2 begin------------------------------------\n");
    // /* 2 test get number of vertices and edges */
    // printf("\ntest get number of vertices and edges\n");
    // printf("number of vertices: %lu, ", g.get_number_of_vertices());
    // printf("number of edges: %lu\n", g.get_number_of_edges());

    // /* 2 test find vertex and edge */
    // printf("\ntest find vertex and edge\n");
    // printf("Is 1 in the graph? %d \n", (int)g.check_vertex(1));
    // printf("Is 1 in the graph? %d \n", (int)g.check_vertex(0));
    // printf("Is (1, 2) in the graph? %d \n", (int)g.check_edge(1, 2));
    // printf("Is (1, 5) in the graph? %d \n", (int)g.check_edge(1, 5));

    // /* 2 test find vertex and edge */
    // printf("\ntest get_weight\n");
    // printf("the value of (1, 2) is %d \n", g.get_weight(1, 2, -1));
    // printf("the value of (1, 5) is %d \n", g.get_weight(1, 5, -1));

    // /* 2 insert vertex */
    // printf("\n test insert vertex\n");
    // printf("insert 0? %d\n", (int)g.insert_vertex(0));
    // printf("insert 1? %d\n", (int)g.insert_vertex(1));
    // printf("----------------------------Test 2 end-------------------------------------\n");

    // printf("----------------------------Test 3 begin------------------------------------\n");
    // /* 3 test get number of in degrees */
    // printf("\ntest get in degrees\n");
    // printf("number of in-degrees for vertex 3: %lu, \n", g.get_in_degree(3));

    // /* 3 test get number of out degrees */
    // printf("\ntest get out degrees\n");
    // printf("number of out-degrees for vertex 1: %lu, \n", g.get_out_degree(1));

    // /* 3 test get number of neighbors */
    // printf("\ntest get number of neighbors\n");
    // printf("number of neighbors for vertex 6: %lu, \n", g.get_num_neighbors(6));

    // printf("----------------------------Test 3 end-------------------------------------\n");

    // printf("----------------------------Test 4 begin------------------------------------\n");
    // printf("\ntest delete vertex\n");

    // size_t v_del = 3;
    // printf("Delete sueccess? %d\n",(int)g.delete_vertex(v_del));

    // /* test print*/
    // g.print();

    // printf("----------------------------Test 4 end-------------------------------------\n");
    // printf("----------------------------Test 5 begin------------------------------------\n");
    // printf("\ntest get destination of vertex\n");

    // printf("\nget destination of 1:\n");
    // std::vector<size_t> destination = g.get_destination_vertex(1);
    // if(destination.empty()) 
    //     printf("No destination vertex for 1.\n");
    // else{
    //     printf("destination of vertex 1: ");
    //     for(int i=0; i < destination.size(); i++) 
    //         printf("%lu\t",destination[i]);
    //     printf("\n");
    // }

    // printf("\ntest get source of vertex\n");
    // std::vector<size_t> source = g.get_source_vertex(1);
    // if(source.empty()) 
    //     printf("No source vertex for 1.\n");
    // else{
    //     for(int i=0; i < source.size(); i++) 
    //         printf("%lu\t",source[i]);
	// }

    // std::vector<size_t> source1 = g.get_source_vertex(6);
    // if(source1.empty()) 
    //     printf("No source vertex for 6.\n");
    // else{
    //     printf("sources of vertex 6: ");
    //     for(int i=0; i < source1.size(); i++) 
    //         printf("%lu\t",source1[i]);
    //     printf("\n");
	// }
    

    /* test print*/
    //g.print();

    //printf("----------------------------Test 5 end-------------------------------------\n");

    return 0;
}
