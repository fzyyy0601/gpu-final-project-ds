#include "graph.hxx"
#include "coo_d.hxx"
#include "coo_h.h"
#include <iostream>
#include <vector>
#include <time.h>
#include <map>
#include <utility>
#include <algorithm>

/* intital graph configuration */
size_t MAX_V; // max number of vertices
const size_t N = 10000000;
size_t MAX; // max number of nodes / memory limit
size_t test_times;   // test times
size_t v_num;       // initial number of vertices
size_t e_num;     // intinal number of edges, e_num should smaller than MAX+test_times
size_t* v_list;       // intinal vertex list
size_t* row_idx;   // initial row index (source of each edge)
size_t* col_idx;   // initial col index (target of each edge)
int* value;        // initial weight for each edge

/* configuration for the kernel call */
size_t number_of_blocks;
size_t threads_per_block;

/* randomly create an intial graph in the host according to 
    number of vertices, edges and max number of vertices */
void init(){
    if(v_num>MAX_V/2){
        std::vector<int> v;
        for(int i=0;i<MAX_V;i++){
            v.push_back(i);
        }
        std::random_shuffle(v.begin(),v.end());
        for(int i=0;i<v_num;i++){
            v_list[i]=v[i];
        }
    }
    else{
        std::map<int,int> mp;
        while(mp.size()<v_num){
            int cx=rand()%MAX_V;
            if(mp.count(cx)){
                continue;
            }
            v_list[mp.size()]=cx;
            mp[cx]=1;
        }
    }
    if(e_num>v_num*v_num/4){
        std::vector<std::pair<int,int> > v;
        for(int i=0;i<v_num;i++){
            for(int j=0;j<v_num;j++){
                v.push_back(std::make_pair(v_list[i],v_list[j]));
            }
        }
        std::random_shuffle(v.begin(),v.end());
        for(int i=0;i<e_num;i++){
            row_idx[i]=v[i].first;
            col_idx[i]=v[i].second;
            value[i]=rand();
        }
    }
    else{
        std::map<std::pair<int,int>,int> mp;
        while(mp.size()<e_num){
            int a=rand()%v_num,b=rand()%v_num,c=rand();
            if(mp.count(std::make_pair(a,b))){
                continue;
            }
            int idx=mp.size();
            row_idx[idx]=v_list[a];
            col_idx[idx]=v_list[b];
            value[idx]=c;
            mp[std::make_pair(a,b)]=c;
        }
    }
}

template<typename weight_t,
template<typename> typename graph_view_t> 
void test(graph<weight_t,graph_view_t> g){

    double time_taken;
    clock_t start, end;
    clock_t T_start, T_end;

    /* create an initial graph in host */
    init();

    printf("\n ----------------------------Test begin------------------------------------\n");

    /* test initialization */
    T_start = clock();
    start = clock();
    g.init(v_list, v_num, row_idx, col_idx, value, e_num, number_of_blocks, threads_per_block, MAX);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("initialization: %.2lf s\n", time_taken);

    /* insert test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.insert_vertex(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("insert %d vertices: %.2lf s\n", test_times, time_taken);

    /* insert test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.insert_edge(rand() % MAX_V, rand() % MAX_V, rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("insert %d edges: %.2lf s\n", test_times, time_taken);

    /* check test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.check_edge(rand() % MAX_V, rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("check %d edges: %.2lf s\n", test_times, time_taken);

    /* check test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.check_vertex(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("check %d vertices: %.2lf s\n", test_times, time_taken);

    /* get weight*/
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_weight(rand() % MAX_V, rand() % MAX_V, -1);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get weight of %d edges: %.2lf s\n", test_times, time_taken);

    /* get in degree of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_in_degree(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get in degree of %d vertices: %.2lf s\n", test_times, time_taken);

    /* get out degree of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_out_degree(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get out degree of %d vertices: %.2lf s\n", test_times, time_taken);

    /* get number of neighbors test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_num_neighbors(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get num of neighbors of %d vertices: %.2lf s\n", test_times, time_taken);

    /* get source of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_source_vertex(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get source of %d vertices: %.2lf s\n", test_times, time_taken);

    /* get destination of test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.get_destination_vertex(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("get destination of %d vertices: %.2lf s\n", test_times, time_taken);

    /* delete test_times edges */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.delete_edge(rand() % MAX_V, rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("delete %d edges: %.2lf s\n", test_times, time_taken);

    /* delete test_times vertices */
    start = clock();
    for (int i = 0; i < test_times; i ++) {
        g.delete_vertex(rand() % MAX_V);
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("delete %d vertices: %.2lf s\n", test_times, time_taken);

    T_end = clock();
    time_taken = ((double)(T_end - T_start))/ CLOCKS_PER_SEC;
    printf("Total time: %.2lf s\n", time_taken);

    // /* test print*/
    // g.print();
    // /* 2 test get number of vertices and edges */
    // printf("\ntest get number of vertices and edges\n");
    // printf("number of vertices: %lu, ", g.get_number_of_vertices());
    // printf("number of edges: %lu\n", g.get_number_of_edges());


    printf("----------------------------Test end---------------------------------------\n");

}

int main(){
    /* set the random seed and clock */
    srand(0);

    /* type in the configuration of the experiment*/
    printf("GPU(0)\\CPU(1) | MAX_V | v_num | e_num | test_times | MAX | grid_size | block_size\n");
    int CGPU;
    scanf("%d%lu%lu%lu%lu%lu",&CGPU,&MAX_V,&v_num,&e_num,&test_times,&MAX);
    
    v_list=(size_t*)malloc(v_num*sizeof(size_t));
    row_idx=(size_t*)malloc(e_num*sizeof(size_t));
    col_idx=(size_t*)malloc(e_num*sizeof(size_t));
    value=(int*)malloc(e_num*sizeof(int));

    /* create an empty graph*/
    if(CGPU==0){
        scanf("%lu%lu",&number_of_blocks,&threads_per_block);
        graph<int,coo_d> g;
        test(g);
    }
    else{
        graph<int,coo_h> g;
        test(g);
    }

    free(v_list);
    free(row_idx);
    free(col_idx);
    free(value);

    // /* print the initial graph we input */
    // printf("The initial graph we input: \n");
    // printf("v_list: ");
    // for(int i = 0; i < v_num; i++){
    //     printf("%lu ", v_list[i]);
    // }
    // printf("\n");
    
    // printf("row_idx: ");
    // for(int i = 0; i < e_num; i++){
    //     printf("%lu ", row_idx[i]);
    // }
    // printf("\n");
    
    // printf("col_idx: ");
    // for(int i = 0; i < e_num; i++){
    //     printf("%lu ", col_idx[i]);
    // }
    // printf("\n");
    
    // printf("value: ");
    // for(int i = 0; i < e_num; i++){
    //     printf("%d ", value[i]);
    // }
    // printf("\n");

    return 0;
}
