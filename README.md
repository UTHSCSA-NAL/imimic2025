Software associated with the publication:
> Nicolas Honnorat, Mohamad Habes, "*Interpretable Networks to Model the Accumulation of Tau Protein in the Brain*", Workshop on Interpretability of Machine Intelligence in Medical Image Computing (iMIMIC 2025)

Run the Bash script training.sh 
to generates cortical maps according to the best results reported in the publication.
These maps will be generated as "maps_left.npy" for the left hemisphere, and "maps_right.npy" for the right hemisphere
and should be similar to "best_maps_left.npy" and "best_maps.right.npy" (edit training.sh to produce different sets of cortical maps)


These maps correspond to the cortical mesh stored in the files:
> Conte69.L.midthickness.32k_fs_LR.surf.gii (left hemisphere) <br>
> Conte69.R.midthickness.32k_fs_LR.surf.gii (right hemisphere)

that can be opened using the nibabel Python library (e.g. for visualization).

The geodesic distances are stored in the "data_XXX_godesic_distances.npy" files, and where calculated for the 22 cortical regions described in the parcellation22.csv file. The cortical parcellation are provided in the files "parcellation22_XXX.txt" (where label 0 correspond to the corpus callosum, a region ignored during the training/geneation of cortical maps).

Similarly, the ADNI tau quantile values for the 5 age groups are stored in the files "data_XXX_quantiles.npy" and were calculated as explained in the publication.
