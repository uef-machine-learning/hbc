#### HaloBasedClustering



HBC: halo-based clustering using local comparative density

version ==1.0

Authors: Le Li (leli@cs.uef.fi)and Fei Wang(feiwang@cs.uef.fi)



#### Dependencies

kneed==0.8.5

matplotlib==3.7.5

numpy==1.24.3

pandas==2.0.3

scikit-learn==1.3.2

scipy==1.10.1

seaborn==0.13.2



#### Parameters

n_neighbors: number of neighbors in k-NN

density_threshold_pct: parameter which ranges between 0 and 100 for determining a relative density threshold for halo detection. 
                   							    density_threshold_pct = 50 is recommended if prior knowledge is absent. 