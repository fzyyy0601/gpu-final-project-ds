#include "graph.hxx"
#include "coo.hxx"
#include "coo_h.h"
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>
#include <stdlib.h>

int main(){
    /* set the random seed and clock */
    int c=time(0);
    std::cout<<sizeof(char)<<" "<<sizeof(size_t)<<" "<<sizeof(bool)<<" "<<sizeof(short)<<" "<<sizeof(int)<<"\n";
//    printf("%lld\n",1ll*sizeof(char));
    printf("RANDSEED c =%d\n",c);
    srand(0);
    for(int i=0;i<10000;i++) std::cout<<rand()<<" ";
    double time_taken;
    clock_t start, end;
    clock_t T_start, T_end;

    /* initialize an empty graph*/
    graph<int,coo> g;
    graph<int,coo_h> gh;
    size_t v_list[] = {};
    size_t v_num = 0;
    size_t row_idx[] = {};
    size_t col_idx[] = {};
    int value[] = {};
    size_t e_num = 0;
    size_t number_of_blocks = 48;
    size_t threads_per_block = 256;
    size_t MAX_V=10000;
    size_t MAX = 1000000;
    size_t test_times = 1000000;
    bool ok=0;

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
    gh.init(v_list, v_num, row_idx, col_idx, value, e_num, number_of_blocks, threads_per_block, MAX);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("initialization: %lf s\n", time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* insert test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int tmp=rand()%MAX_V;
        g.insert_vertex(tmp);
        gh.insert_vertex(tmp);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("insert %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* insert test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V,b=rand()%MAX_V,c=rand()%MAX_V;
        g.insert_edge(a,b,c);
        gh.insert_edge(a,b,c);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("insert %d edges: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* check test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V,b=rand()%MAX_V;
        if(g.check_edge(a, b)!=gh.check_edge(a, b)){
            if(!ok) printf("NOT %d %d\n",a,b);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("check %d edges: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* check test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V;
        if(g.check_vertex(a)!=gh.check_vertex(a)){
            if(!ok)printf("NOT %d\n",a);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("check %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get weight*/
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V,b=rand()%MAX_V;
        if(g.get_weight(a,b, -1)!=gh.get_weight(a,b, -1)){
            if(!ok)printf("NOT %d %d\n",a,b);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get weight of %d edges: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get in degree of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V;
        if(g.get_in_degree(a)!=gh.get_in_degree(a)){
            printf("%d %d\n",g.get_in_degree(a),gh.get_in_degree(a));
            if(!ok)printf("NOT %d\n",a);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get in degree of %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get out degree of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand() % MAX_V;
        if(g.get_out_degree(a)!=gh.get_out_degree(a)){
            if(!ok)printf("NOT %d\n",a);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get out degree of %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get number of neighbors test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand() % MAX_V;
        if(g.get_num_neighbors(a)!=gh.get_num_neighbors(a)){
            if(!ok)printf("NOT %d\n",a);ok=1;
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get num of neighbors of %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get source of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand() % MAX_V;
        std::vector<size_t> v1=g.get_source_vertex(a);
        std::vector<size_t> v2=gh.get_source_vertex(a);
        std::sort(v1.begin(),v1.end());
        std::sort(v2.begin(),v2.end());
        if(v1!=v2){
            for(int i=0;i<v1.size();i++) printf("%d ",v1[i]);printf("\n");
            for(int i=0;i<v2.size();i++) printf("%d ",v2[i]);printf("\n");
            if(!ok)printf("NOT %d\n",a);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get source of %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* get destination of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand() % MAX_V;
        std::vector<size_t> v1=g.get_destination_vertex(a);
        std::vector<size_t> v2=gh.get_destination_vertex(a);
        std::sort(v1.begin(),v1.end());
        std::sort(v2.begin(),v2.end());
        if(v1!=v2){
            for(int i=0;i<v1.size();i++) printf("%d ",v1[i]);printf("\n");
            for(int i=0;i<v2.size();i++) printf("%d ",v2[i]);printf("\n");
            if(!ok)printf("NOT %d\n",a);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get destination of %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* delete test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V,b=rand()%MAX_V;
        if(g.delete_edge(a,b)!=gh.delete_edge(a,b)){
            if(!ok)printf("NOT %d %d\n",a,b);ok=1; 
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("delete %d edges: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    /* delete test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        int a=rand()%MAX_V;
        if(g.delete_vertex(a)!=gh.delete_vertex(a)){
            if(!ok)printf("NOT\n");ok=1;
        }
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("delete %d vertices: %lf s\n", test_times, time_taken);

    g.print_config();
    gh.print_config();
    if(ok) {printf("3rwiejf vndms,qwerjfibgh cxaskwoieh vcxkslawiedh vcxkswerhybfvsdwerfvnwrefbvnswenrbfgswenrfvdcsqweorcvksweoi3svfsweirbfgswercbfgnjrejithcfkjrei49tgbfjedrut5gedo3irtg");ok=0;}

    T_end = clock();
    time_taken = ((double)(T_end - T_start))/ CLOCKS_PER_SEC;
    printf("Total time: %lf s\n", time_taken);

    // /* test print*/
    // g.print();

    // /* 2 test get number of vertices and edges */
    // printf("\ntest get number of vertices and edges\n");
    // printf("number of vertices: %lu, ", g.get_number_of_vertices());
    // printf("number of edges: %lu\n", g.get_number_of_edges());


    printf("----------------------------Test end---------------------------------------\n");

    return 0;
}
