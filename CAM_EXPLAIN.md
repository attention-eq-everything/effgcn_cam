# Explaining CAM

## Part 1: Understand Processor

![Processor code explanation](assets/processor_required_vars.png)

Please refer to the image given above.
It basically explains how the file `visualization/extraction_{config_name}.npz` is created
and what all does it contain. All the data written into this `npz` file is required for
visualizing the NTU skeletons with the joint importances obtained at the final layer after
a series of layers having `ST-JointAtt` module.


## Part 2: Network's layers

![Network layers used](assets/nets_layers.png)

This part explains the layers used in the network which actually are important and where they are located
in the network structure. It also explains the purpose of the ST-JointAtt module.


## Part 3: Spatial Temporal Joint Attention

![ST Joint Attention Paper Figure](resources/st_joint_att.jpg)

![ST Joint Attention Module](assets/st_jointatt_explain.png)

This part explains the actual Spatial Temporal Joint Attention module code and how it actually works.
This is main module used in attending over the joints in current frame as well as across the temporal dimension.
Importances from this module actually contribute a lot to the final importances obtained for visualization.


## Part 4: Visualization code

![Visualizer code](assets/visualizer_final.png)

This part explains the code in the visualizer that calculates the required vectors which can be used for plotting
the NTU skeletons and their importances in an interpretable manner.