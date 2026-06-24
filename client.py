import argparse
import flwr as fl

from fl_common import (
    create_models,
    get_combined_parameters,
    train_local_client,
    DEVICE,
    LOCAL_EPOCHS,
)


class Pix2PixClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, direction_name: str):
        self.client_id = client_id
        self.direction_name = direction_name

    def get_parameters(self, config):
        print(f"[CLIENT {self.client_id}] get_parameters called", flush=True)
        netG, netD = create_models(DEVICE)
        return get_combined_parameters(netG, netD)

    def fit(self, parameters, config):
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        server_round = config.get("server_round", "NA")

        print(
            f"[CLIENT {self.client_id}] fit called | round={server_round} | epochs={local_epochs}",
            flush=True,
        )

        updated_parameters, num_examples, metrics = train_local_client(
            direction_name=self.direction_name,
            client_id=self.client_id,
            parameters=parameters,
            local_epochs=local_epochs,
        )

        print(
            f"[CLIENT {self.client_id}] fit finished | "
            f"num_examples={num_examples} | "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"val_ssim={metrics['val_ssim']:.6f}",
            flush=True,
        )

        return updated_parameters, num_examples, {
            "client_id": int(self.client_id),
            "val_loss": float(metrics["val_loss"]),
            "val_l1": float(metrics["val_l1"]),
            "val_ssim": float(metrics["val_ssim"]),
            "val_psnr": float(metrics["val_psnr"]),
        }

    def evaluate(self, parameters, config):
        print(f"[CLIENT {self.client_id}] evaluate skipped", flush=True)
        return 0.0, 0, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--direction", type=str, default="T2_to_T1")
    parser.add_argument("--grpc_max_message_length", type=int, default=2147483647)

    args = parser.parse_args()

    print(
        f"[CLIENT {args.client_id}] Connecting to {args.server_address} | direction={args.direction}",
        flush=True,
    )

    client = Pix2PixClient(
        client_id=args.client_id,
        direction_name=args.direction,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
        grpc_max_message_length=args.grpc_max_message_length,
    )


if __name__ == "__main__":
    main()
