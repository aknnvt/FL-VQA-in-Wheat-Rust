import flwr as fl
from flwr.server.strategy import FedAvg
from central_model import CentralModel

# Define Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config={"num_rounds": 10},
    strategy=FedAvg(min_available_clients=2),
    client_manager=fl.server.client_manager.SimpleClientManager(
        clients=["client1", "client2"]
    ),
    central_model=CentralModel()
)