#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pickle

def main():
    # train_instance = dl.load_instance('train_class')

    with open("../Deep_Learning/Neural_Network/saveNetwork/train_class"+".pkl","rb") as file:
        save_instance = pickle.load(file)
    print(save_instance.__class__.__name__)
    print("load success")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HabiWeb.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"

        ) from exc
    execute_from_command_line(sys.argv)
    # train_instance.train_step()


if __name__ == '__main__':
    main()
