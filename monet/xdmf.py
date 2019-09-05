import os
import h5py

import numpy as np


def z_file_to_xdmf(z_file, xdmf_dir, filename):
    xdmf_file = os.path.join(xdmf_dir, "%s.xmf" % filename)
    h5_file_name = "%s.h5" % filename
    h5_file = os.path.join(xdmf_dir, h5_file_name)

    with h5py.File(z_file, "r") as fh5:
        z_data = np.array(fh5["z"])
        gx_data = np.array(fh5["gx"])
        gy_data = np.array(fh5["gy"])

    # Dimensions
    nx, ny = z_data.shape[0], z_data.shape[1]

    # Coordinates
    X = np.arange(0, nx)
    Y = np.arange(0, ny)

    x_2d, y_2d = np.meshgrid(X, Y)

    x = np.zeros((nx, ny), dtype='float32')
    x[:, :] = x_2d
    y = np.zeros((nx, ny), dtype='float32')
    y[:, :] = y_2d
    z = np.zeros((nx, ny), dtype='float32')
    z[:, :] = z_data
    gx = np.zeros((nx, ny), dtype='float32')
    gx[:, :] = gx_data
    gy = np.zeros((nx, ny), dtype='float32')
    gy[:, :] = gy_data

    with h5py.File(h5_file, 'w') as h5:
        h5.create_dataset("X", data=x)
        h5.create_dataset("Y", data=y)
        h5.create_dataset("Z", data=z)
        h5.create_dataset("GX", data=gx)
        h5.create_dataset("GY", data=gy)

    with open(xdmf_file, "w") as fout:
        shape = " ".join(map(str, z.shape))
        fout.write("""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="mesh1" GridType="Uniform">
     <Topology TopologyType="2DSMesh" NumberOfElements="%s"/>
     <Geometry GeometryType="X_Y">
       <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">
        %s:/X
       </DataItem>
       <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">
        %s:/Y
       </DataItem>
     </Geometry>
     <Attribute Name="Z" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">
        %s:/Z
       </DataItem>
     </Attribute>
     <Attribute Name="GradX" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">
        %s:/GX
       </DataItem>
     </Attribute>
     <Attribute Name="GradY" AttributeType="Scalar" Center="Node">
       <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">
        %s:/GY
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
            """ % (shape, shape, h5_file_name, shape, h5_file_name,
                   shape, h5_file_name, shape, h5_file_name, shape, h5_file_name))
    return xdmf_file, h5_file