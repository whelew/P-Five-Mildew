from streamlit_pages.multipage import MultiPage
from streamlit_pages import (
    page_hypothesis,
    page_leaf_visualiser,
    page_ml_performance,
    page_powdery_mildew_detector,
    page_summary
)

app = MultiPage(app_name='Mildew Detection Within Cherry Leaves')

app.add_page('Project Summary', page_summary.page_summary)
app.add_page('Project Hypothesis', page_hypothesis.page_hypothesis)
app.add_page('Leaf Visualisation', page_leaf_visualiser.page_leaf_visualiser_body)
app.add_page('Machine Learning Performance Metrics', page_ml_performance.page_ml_performance_metrics)
app.add_page('Powdery Mildew Detector', page_powdery_mildew_detector.page_powdery_mildew_detector)

app.run()
