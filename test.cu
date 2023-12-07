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
    std::cout<<g.get_number_of_vertices();
    // g.get_number_of_vertices<<<1,1>>>();
    int a;
    return 0;
}
