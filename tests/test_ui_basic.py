"""
Basic UI tests for VibroVis application.
Tests that the app loads and main components are visible.
"""
import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_app_loads(app_page):
    """Test that the application loads successfully."""
    # Check page title
    assert "VibroVis" in app_page.title()
    
    # Check that the page loaded without errors
    # Look for the main container
    main_container = app_page.locator('#main-container')
    expect(main_container).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_scatter_plot_visible(app_page):
    """Test that the scatter plot is visible."""
    scatter_plot = app_page.locator('#scatter')
    expect(scatter_plot).to_be_visible(timeout=10000)
    
    # Check that it's a graph element
    scatter_container = app_page.locator('#scatter-audio-container')
    expect(scatter_container).to_be_visible()


@pytest.mark.ui
def test_filter_panel_visible(app_page):
    """Test that the filter panel is visible."""
    filter_panel = app_page.locator('#filter-panel-container')
    expect(filter_panel).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_location_dropdown_exists(app_page):
    """Test that the location dropdown exists."""
    location_dropdown = app_page.locator('#location-dropdown')
    expect(location_dropdown).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_model_dropdown_exists(app_page):
    """Test that the model dropdown exists."""
    model_dropdown = app_page.locator('#model-dropdown')
    expect(model_dropdown).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_num_cluster_dropdown_exists(app_page):
    """Test that the number of clusters dropdown exists."""
    num_cluster_dropdown = app_page.locator('#num-cluster-dropdown')
    expect(num_cluster_dropdown).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_histogram_visible(app_page):
    """Test that the histogram container is visible."""
    histogram_container = app_page.locator('#histogram-container')
    expect(histogram_container).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_audio_player_exists(app_page):
    """Test that the audio player element exists."""
    audio_player = app_page.locator('#audio-player')
    expect(audio_player).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_spectrogram_plot_exists(app_page):
    """Test that the spectrogram plot container exists."""
    spectrogram_container = app_page.locator('#spectrogram-plot-container')
    expect(spectrogram_container).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_preset_dropdown_exists(app_page):
    """Test that the preset dropdown exists."""
    preset_dropdown = app_page.locator('#preset-load-dropdown')
    expect(preset_dropdown).to_be_visible(timeout=10000)


@pytest.mark.ui
def test_autoplay_button_exists(app_page):
    """Test that the autoplay toggle button exists."""
    autoplay_button = app_page.locator('#autoplay-toggle-btn')
    expect(autoplay_button).to_be_visible(timeout=10000)
