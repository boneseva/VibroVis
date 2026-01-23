import cProfile
import pstats
from app import app  # Import your app object


def start():
    # We try to initialize everything manually to see where it sticks
    import read_data
    from callbacks import set_initial_data, register_callbacks
    from layout import create_layout

    print("Loading data...")
    df = read_data.get_initial_data_for_layout()
    print("Setting data...")
    set_initial_data(df)
    print("Creating layout...")
    app.layout = create_layout(df)
    print("Registering callbacks...")
    register_callbacks(app)
    print("Done! If you see this, the hang is inside app.run()")


if __name__ == "__main__":
    # This will run your startup and save the results to 'output.prof'
    cProfile.run('start()', 'output.prof')

    # Print the results to the terminal immediately
    p = pstats.Stats('output.prof')
    p.sort_stats('cumulative').print_stats(30)