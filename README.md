
# Inspiration:
https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/?utm_source=golangweekly&utm_medium=email


# install go-bindings
Follow the instructions on
https://www.tensorflow.org/install/install_go

## TLDR;
Do this: ( if anything fails, find out why on https://www.tensorflow.org/install/install_go )

TF_TYPE="cpu" # Change to "gpu" for GPU support
TARGET_DIRECTORY='/usr/local'
curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.1.0.tar.gz" |
sudo tar -C $TARGET_DIRECTORY -xz

sudo ldconfig

go get github.com/tensorflow/tensorflow/tensorflow/go
go test github.com/tensorflow/tensorflow/tensorflow/go

# generate a model to save

you need to build a builder with tags:

    builder = tf.saved_model.builder.SavedModelBuilder("export2")
    
    builder.add_meta_graph_and_variables(sess,["serve"])
    
    builder.save()

# load the model
