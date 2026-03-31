from openenv.core.env_server import create_app
from server.inventory_env import InventoryEnvironment
from models import InventoryAction, InventoryObservation

app = create_app(InventoryEnvironment, InventoryAction, InventoryObservation, env_name="inventory_env")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()