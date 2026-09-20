import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# USED CAR PRICE PREDICTION
# SIMPLE + ADVANCED CLI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "car_price_random_forest.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "car_price_preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "car_price_metadata.joblib")


# These are the same feature groups used during training.
NUMERICAL_FEATURES = [
    "km",
    "No of Cylinder",
    "Valves per Cylinder",
    "Length",
    "Width",
    "Height",
    "Wheel Base",
    "Kerb Weight",
    "Seats",
    "Doors",
    "Cargo Volume",
    "Max Power Delivered",
    "Max Power At",
    "Max Torque Delivered",
    "Max Torque At",
    "vehicle_age"
]

CATEGORICAL_FEATURES = [
    "body",
    "transmission",
    "fuel",
    "oem",
    "model",
    "owner_type",
    "state",
    "Gear Box",
    "Drive Type",
    "Steering Type",
    "Front Brake Type",
    "Rear Brake Type",
    "Tyre Type",
    "Turbo Charger",
    "Super Charger",
    "Valve Configuration"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ask_number(prompt, integer=False, minimum=None, maximum=None, allow_blank=True):
    """Read and validate a numeric input."""

    while True:
        value = input(prompt).strip()

        if value == "":
            if allow_blank:
                return np.nan
            print("This value is required.")
            continue

        try:
            number = int(value) if integer else float(value)
        except ValueError:
            print("Invalid value. Please enter a number.")
            continue

        if minimum is not None and number < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        if maximum is not None and number > maximum:
            print(f"Value must not be greater than {maximum}.")
            continue

        return number


def ask_text(prompt, allow_blank=True):
    """Read a text input."""

    while True:
        value = input(prompt).strip()

        if value == "" and not allow_blank:
            print("This value is required.")
            continue

        return np.nan if value == "" else value.lower()


def format_price(price):
    """Format Indian Rupee price."""

    return f"₹{price:,.2f}"


def load_artifacts():
    """Load the saved model, preprocessor and metadata."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Random Forest model not found:\n{MODEL_PATH}\n"
            "Run preprocessing_final.py first."
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}\n"
            "Run preprocessing_final.py first."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)

    return model, preprocessor, metadata


def make_prediction(user_data, model, preprocessor):
    """Create a complete feature row and predict the price."""

    # Create all expected features.
    # Missing technical values are intentionally left as NaN so that
    # the SAME saved imputer used during training handles them.
    row = {feature: np.nan for feature in ALL_FEATURES}

    for key, value in user_data.items():
        if key in row:
            row[key] = value

    input_df = pd.DataFrame([row], columns=ALL_FEATURES)

    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]

    return max(0, float(prediction))


# ============================================================
# SIMPLE PREDICTION
# ============================================================

def simple_prediction(model, preprocessor, metadata):
    print("\n========================================")
    print("       SIMPLE USED CAR PREDICTOR")
    print("========================================")
    print("Enter the main vehicle details.")
    print("Technical specifications can be left blank.")
    print()

    reference_year = metadata.get("reference_year", 2024)

    # Required/important user-friendly inputs.
    model_year = ask_number(
        "Model Year: ",
        integer=True,
        minimum=1980,
        maximum=reference_year,
        allow_blank=False
    )

    km = ask_number(
        "Kilometers Driven: ",
        minimum=0,
        allow_blank=False
    )

    body = ask_text(
        "Body Type (SUV/Sedan/Hatchback/etc.): "
    )

    transmission = ask_text(
        "Transmission (manual/automatic): "
    )

    fuel = ask_text(
        "Fuel (petrol/diesel/cng/etc.): "
    )

    oem = ask_text(
        "Brand/OEM (e.g. hyundai/maruti/tata): "
    )

    model_name = ask_text(
        "Car Model (e.g. creta/swift/nexon): "
    )

    owner_type = ask_text(
        "Owner Type (e.g. first owner/second owner): "
    )

    state = ask_text(
        "State (e.g. maharashtra/delhi): "
    )

    # Training used:
    # vehicle_age = reference_year - myear
    vehicle_age = reference_year - int(model_year)

    user_data = {
        "km": km,
        "vehicle_age": vehicle_age,
        "body": body,
        "transmission": transmission,
        "fuel": fuel,
        "oem": oem,
        "model": model_name,
        "owner_type": owner_type,
        "state": state,
    }

    prediction = make_prediction(
        user_data,
        model,
        preprocessor
    )

    print("\n========================================")
    print("       ESTIMATED USED CAR PRICE")
    print("========================================")
    print(f"Estimated Price: {format_price(prediction)}")
    print(f"Approx. Price:   ₹{prediction / 100000:.2f} Lakh")
    print("========================================")

    print("\nNote:")
    print("This is an ML estimate based on the trained dataset.")
    print("It is not a guaranteed market price.")


# ============================================================
# ADVANCED PREDICTION
# ============================================================

def advanced_prediction(model, preprocessor, metadata):
    print("\n========================================")
    print("       ADVANCED USED CAR PREDICTOR")
    print("========================================")
    print("Press Enter to leave a value missing.")
    print("The saved preprocessor will impute missing values.")
    print()

    reference_year = metadata.get("reference_year", 2024)

    model_year = ask_number(
        "Model Year: ",
        integer=True,
        minimum=1980,
        maximum=reference_year,
        allow_blank=False
    )

    user_data = {
        "km": ask_number("km: ", minimum=0),
        "No of Cylinder": ask_number("No of Cylinder: ", minimum=1),
        "Valves per Cylinder": ask_number("Valves per Cylinder: ", minimum=1),
        "Length": ask_number("Length: ", minimum=0),
        "Width": ask_number("Width: ", minimum=0),
        "Height": ask_number("Height: ", minimum=0),
        "Wheel Base": ask_number("Wheel Base: ", minimum=0),
        "Kerb Weight": ask_number("Kerb Weight: ", minimum=0),
        "Seats": ask_number("Seats: ", minimum=1),
        "Doors": ask_number("Doors: ", minimum=1),
        "Cargo Volume": ask_number("Cargo Volume: ", minimum=0),
        "Max Power Delivered": ask_number("Max Power Delivered: ", minimum=0),
        "Max Power At": ask_number("Max Power At: ", minimum=0),
        "Max Torque Delivered": ask_number("Max Torque Delivered: ", minimum=0),
        "Max Torque At": ask_number("Max Torque At: ", minimum=0),
        "body": ask_text("body: "),
        "transmission": ask_text("transmission: "),
        "fuel": ask_text("fuel: "),
        "oem": ask_text("oem: "),
        "model": ask_text("model: "),
        "owner_type": ask_text("owner_type: "),
        "state": ask_text("state: "),
        "Gear Box": ask_text("Gear Box: "),
        "Drive Type": ask_text("Drive Type: "),
        "Steering Type": ask_text("Steering Type: "),
        "Front Brake Type": ask_text("Front Brake Type: "),
        "Rear Brake Type": ask_text("Rear Brake Type: "),
        "Tyre Type": ask_text("Tyre Type: "),
        "Turbo Charger": ask_text("Turbo Charger: "),
        "Super Charger": ask_text("Super Charger: "),
        "Valve Configuration": ask_text("Valve Configuration: "),
    }

    user_data["vehicle_age"] = reference_year - int(model_year)

    prediction = make_prediction(
        user_data,
        model,
        preprocessor
    )

    print("\n========================================")
    print("       ESTIMATED USED CAR PRICE")
    print("========================================")
    print(f"Estimated Price: {format_price(prediction)}")
    print(f"Approx. Price:   ₹{prediction / 100000:.2f} Lakh")
    print("========================================")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("\n========================================")
    print("       USED CAR PRICE PREDICTION")
    print("========================================")

    try:
        model, preprocessor, metadata = load_artifacts()
    except Exception as error:
        print("\nERROR:")
        print(error)
        return

    print("Saved model loaded successfully.")
    print("Saved preprocessor loaded successfully.")

    while True:
        print("\n========================================")
        print("             SELECT MODE")
        print("========================================")
        print("1. Simple Prediction")
        print("2. Advanced Prediction")
        print("3. Exit")
        print("========================================")

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            try:
                simple_prediction(model, preprocessor, metadata)
            except Exception as error:
                print("\nPrediction error:")
                print(error)

        elif choice == "2":
            try:
                advanced_prediction(model, preprocessor, metadata)
            except Exception as error:
                print("\nPrediction error:")
                print(error)

        elif choice == "3":
            print("\nThank you for using Used Car Price Prediction!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
