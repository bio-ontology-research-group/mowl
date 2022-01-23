<p align="center">
  <img src="docs/source/mowl_black_background_colors_2048x2048px.png" width="600"/>
</p>
  
# Machine Learning with Ontologies
## Library and tool for machine learning with OWL ontologies


### Development Setup

```
conda env create -f environment.yml
conda activate mowl

mkdir -p ../data

cd mowl
./rebuild.sh

```

## Docker

```bash
docker build --tag mowl .
docker
docker run --mount type=bind,source="$(pwd)"/,target=/home/mowl -it -p 8888:8888 --name mowl mowlconda
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

## Documentation

To read our docs please click [here](https://mowl.readthedocs.io/en/latest/index.html)
