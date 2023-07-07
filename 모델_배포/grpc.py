import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

# gRPC 채널 설정, 지정된 호스트 주소와 포트를 사용해 gRPC 서버 연결 제공
def create_grpc_stub(host, port=8500):
    hostport = f"{host}:{port}"
    channel = grpc.insecure_channel(hostport)
    # stub은 사용할 수 있는 메서드를 서버에서 복제한 객체
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def grpc_request(stub, data_sample, model_name='my_model', signature_name='classification'):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name

    # inputs은 신경 네트워크의 입력 이름
    request.inputs['inputs'].CopyFrom(
        tf.make_tensor_proto(data_sample, shape=[1, 1]))

    # 10은 피처가 시간 초과되기 전 최대 시간(초)를 나타냄
    result_future = stub.Predict.future(request, 10)
    return result_future


stub = create_grpc_stub('localhost', port=8500)
rs_grpc = grpc_request(stub, data)
