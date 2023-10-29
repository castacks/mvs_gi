FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install colored polygraphy onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com

WORKDIR /workspace