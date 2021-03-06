#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
#include <stdint.h>
#include <math.h>
using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	    const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const int& ind1, const int& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};

/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
void malis_loss_weights_cpp(const int nVert, const int64_t* seg,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               uint64_t* nPosPairPerEdge, uint64_t* nNegPairPerEdge,
			   bool ignore_background = true,
			   int counting_method = 0,
			   int stochastic_malis_parameter = 0){

    /* Disjoint sets and sparse overlap vectors */
    vector<map<int,uint64_t> > overlap(nVert);
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i){
        dsets.make_set(i);
		if (ignore_background == true) {
			if (seg[i]!=0){
				overlap[i].insert(pair<int,uint64_t>(seg[i],1));
			}
		} else {
			overlap[i].insert(pair<int,uint64_t>(seg[i],1));
		}
    }

    /* Sort all the edges in increasing order of weight */
    std::vector< int > pqueue( nEdge );
    int j = 0;
    for ( int i = 0; i < nEdge; i++ ){
        if ((node1[i]>=0) && (node1[i]<nVert) && (node2[i]>=0) && (node2[i]<nVert)){
			pqueue[ j++ ] = i;
		}
	
    }
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edgeWeight ) );

    /* make the algorithm non-greedy by shuffling the edge list slightly
	 * (list is shuffled more with bigger stochastic_malis_parameter)
	 * */
    if (stochastic_malis_parameter > 0){
		auto lambda_myrandom = [stochastic_malis_parameter] (int i) {
			int rand_max = stochastic_malis_parameter;
			if (i < rand_max){
				return std::rand()%i;
			} else {
				return std::rand()%rand_max;
			}
		};
		std::random_shuffle(pqueue.begin(), pqueue.end(), lambda_myrandom);
	}


    /* Start MST */
    int e;
    int set1, set2;
    uint64_t nPair = 0;
    map<int,uint64_t>::iterator it1, it2;

    /* Start Kruskal's */
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
        e = pqueue[i];

        set1 = dsets.find_set(node1[e]);
        set2 = dsets.find_set(node2[e]);

        if (set1!=set2){
            dsets.link(set1, set2);

            /* compute the number of pairs merged by this MST edge */
            for (it1 = overlap[set1].begin();
                    it1 != overlap[set1].end(); ++it1) {
                for (it2 = overlap[set2].begin();
                        it2 != overlap[set2].end(); ++it2) {
                    
					if (counting_method == 0){
						nPair = it1->second * it2->second;
					} else if (counting_method == 1){
						nPair = log(it1->second) * it2-> second + it1->second * log(it2->second);
					}else if (counting_method == 2){
						nPair = it1->second + it2->second;
					}

                    if ((it1->first == it2->first) && (it1->first !=0)&&(it2->first!=0)) {
						/* only count positives pair that are not background voxels */
						nPosPairPerEdge[e] += nPair;
                    } else {
                        nNegPairPerEdge[e] += nPair;
                    }
                }
            }

            /* move the pixel bags of the non-representative to the representative */
            if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
                swap(set1,set2);

            it2 = overlap[set2].begin();
            while (it2 != overlap[set2].end()) {
                it1 = overlap[set1].find(it2->first);
                if (it1 == overlap[set1].end()) {
                    overlap[set1].insert(pair<int,uint64_t>(it2->first,it2->second));
                } else {
                    it1->second += it2->second;
                }
                overlap[set2].erase(it2++);
            }
        } // end link

    } // end while
}


void connected_components_cpp(const int nVert,
               const int nEdge, const int* node1, const int* node2, const int* edgeWeight,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i)
        dsets.make_set(i);

    /* union */
    for (int i = 0; i < nEdge; ++i )
         // check bounds to make sure the nodes are valid
        if ((edgeWeight[i]!=0) && (node1[i]>=0) && (node1[i]<nVert) && (node2[i]>=0) && (node2[i]<nVert))
            dsets.union_set(node1[i],node2[i]);

    /* find */
    for (int i = 0; i < nVert; ++i)
        seg[i] = dsets.find_set(i);
}


void marker_watershed_cpp(const int nVert, const int* marker,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i)
        dsets.make_set(i);

    /* initialize output array and find representatives of each class */
    std::map<int,int> components;
    for (int i=0; i<nVert; ++i){
        seg[i] = marker[i];
        if (seg[i] > 0)
            components[seg[i]] = i;
    }

    // merge vertices labeled with the same marker
    for (int i=0; i<nVert; ++i)
        if (seg[i] > 0)
            dsets.union_set(components[seg[i]],i);

    /* Sort all the edges in decreasing order of weight */
    std::vector<int> pqueue( nEdge );
    int j = 0;
    for (int i = 0; i < nEdge; ++i)
        if ((edgeWeight[i]!=0) &&
            (node1[i]>=0) && (node1[i]<nVert) &&
            (node2[i]>=0) && (node2[i]<nVert) &&
            (marker[node1[i]]>=0) && (marker[node2[i]]>=0))
                pqueue[ j++ ] = i;
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edgeWeight ) );

    /* Start MST */
	int e;
    int set1, set2, label_of_set1, label_of_set2;
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
		e = pqueue[i];
        set1=dsets.find_set(node1[e]);
        set2=dsets.find_set(node2[e]);
        label_of_set1 = seg[set1];
        label_of_set2 = seg[set2];

        if ((set1!=set2) &&
            ( ((label_of_set1==0) && (marker[set1]==0)) ||
             ((label_of_set2==0) && (marker[set1]==0))) ){

            dsets.link(set1, set2);
            // either label_of_set1 is 0 or label_of_set2 is 0.
            seg[dsets.find_set(set1)] = std::max(label_of_set1,label_of_set2);
            
        }

    }

    // write out the final coloring
    for (int i=0; i<nVert; i++)
        seg[i] = seg[dsets.find_set(i)];

}
