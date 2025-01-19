import h5py 

file = '/net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/mistral_checkpoints/h5_files/output_chunk_0.h5'

with h5py.File(file, 'r') as file:
    data = file['data'][:] 
    print(data)

