import json
import os
from typing import TYPE_CHECKING

import napari
import numpy as np
import torch
from magicgui.widgets import Button, CheckBox, create_widget
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .utils import (
    build_post_transform,
    downsize_window,
    extract_tumor_window,
    full_scan_normalize,
    insert_patch,
    load_model_hf,
    scan_hf_repo,
    single_image_prediction,
    up_sample_pred,
)

if TYPE_CHECKING:
    import napari

MODEL_WIN_SIZE = 64


class MouseTumourAnnotationQWidget(QWidget):
    def __init__(self, napari_viewer, models_dir="models"):
        super().__init__()
        self.viewer = napari_viewer

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.layout = QVBoxLayout()

        self.setLayout(self.layout)

        # Make scrollable
        self.scroll = QScrollArea(self)
        self.layout.addWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget(self.scroll)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_content.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
        )

        # Inference
        self.infer_gb = QGroupBox("Inference")
        self.infer_gb_layout = QGridLayout()
        self.infer_gb.setLayout(self.infer_gb_layout)

        # self.models_dir = models_dir
        # available_models = os.listdir(self.models_dir)
        available_models = scan_hf_repo()

        self.model_cb = QComboBox()
        self.model_cb.addItems(available_models)
        self.model_cb.currentTextChanged.connect(self._on_model_cb_changed)

        self.img_layer_cb = create_widget(annotation=napari.layers.Image)
        self.viewer.layers.events.inserted.connect(
            self.img_layer_cb.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.img_layer_cb.reset_choices
        )
        self.img_layer_cb.changed.connect(self._on_img_layer_changed)

        self.winsize_cb = QComboBox()
        self.winsize_cb.addItem("default", 1)
        self.winsize_cb.addItem("x2", 2)
        self.winsize_cb.addItem("x3", 3)
        self.winsize_cb.currentIndexChanged.connect(
            lambda index: self._on_winsize_cb_change(
                self.winsize_cb.itemData(index)
            )
        )

        self.run_bt = Button(label="RUN")
        self.run_bt.clicked.connect(self._on_run_bt)

        self.infer_gb_layout.addWidget(QLabel("Model"), 0, 0)
        self.infer_gb_layout.addWidget(self.model_cb, 0, 1, 1, 3)
        self.infer_gb_layout.addWidget(QLabel("Image"), 1, 0)
        self.infer_gb_layout.addWidget(self.img_layer_cb.native, 1, 1, 1, 3)
        self.infer_gb_layout.addWidget(QLabel("Window Size"), 2, 0)
        self.infer_gb_layout.addWidget(self.winsize_cb, 2, 1, 1, 3)
        self.infer_gb_layout.addWidget(self.run_bt.native, 3, 0, 1, 4)

        # Merge
        self.merge_gb = QGroupBox("Merge Results")
        self.merge_gb_layout = QGridLayout()
        self.merge_gb.setLayout(self.merge_gb_layout)

        plugin_layer_labels = {"Prediction", "Window Display"}
        self.label_layer_cb = create_widget(
            annotation=napari.layers.Labels,
            options={
                "choices": lambda widget: [
                    layer
                    for layer in self.viewer.layers
                    if isinstance(layer, napari.layers.Labels)
                    and layer.name not in plugin_layer_labels
                ]
            },
        )
        self.viewer.layers.events.inserted.connect(
            self.label_layer_cb.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.label_layer_cb.reset_choices
        )
        self.label_layer_cb.changed.connect(self._on_label_layer_changed)

        self.target_sb = self.bin_size_spinbox = create_widget(
            annotation=int, options={"min": 1, "max": 1000}
        )

        self.merge_bt = Button(label="ADD")
        self.merge_bt.clicked.connect(self._on_merge_bt)

        # self.merge_overwrite_checkbox = CheckBox(value=False, text="Overwrite")

        self.merge_gb_layout.addWidget(QLabel("Prediction"), 0, 0)
        self.merge_gb_layout.addWidget(self.label_layer_cb.native, 0, 1, 1, 3)
        self.merge_gb_layout.addWidget(QLabel("Target Label"), 1, 0)
        self.merge_gb_layout.addWidget(self.target_sb.native, 1, 1, 1, 3)
        # self.merge_gb_layout.addWidget(
        #     self.merge_overwrite_checkbox.native, 1, 3, 1, 2
        # )
        self.merge_gb_layout.addWidget(self.merge_bt.native, 2, 0, 1, 4)

        # Assembly
        self.scroll_layout.addWidget(self.infer_gb)
        self.scroll_layout.addWidget(self.merge_gb)

        # Non widget stuff
        self.window_display_layer = None
        self.prediction_layer = None
        self._on_label_layer_changed(self.label_layer_cb.value)
        self._on_img_layer_changed(self.img_layer_cb.value)
        self._on_model_cb_changed(self.model_cb.currentText())

        self.destroyed.connect(self._on_close)

        self.viewer.mouse_drag_callbacks.append(self._on_click)

        self.center_coords = None
        self.winsize = MODEL_WIN_SIZE
        self.norm_img = None

    def _on_model_cb_changed(self, model_name):
        self.model = load_model_hf(model_name, self.device)

        self.post_transform = build_post_transform()

    def _on_img_layer_changed(self, layer):
        self.refresh_plugin_display_layers()

        if layer is not None:
            self.norm_img = full_scan_normalize(layer.data)
        else:
            self.norm_img = None

    def _on_winsize_cb_change(self, value):
        self.winsize = int(MODEL_WIN_SIZE * value)

        if self.center_coords is not None:
            self.update_window_display(self.center_coords, self.winsize // 2)

    def _on_click(self, viewer, event):
        if event.button != 2:
            return

        img_layer = self.img_layer_cb.value
        if img_layer is None:
            return
        if self.norm_img is None:
            self.norm_img = full_scan_normalize(img_layer.data)

        coords = tuple(
            int(c) for c in img_layer.world_to_data(viewer.cursor.position)
        )

        self.center_coords = coords
        if not all(
            0 <= c < s
            for c, s in zip(coords, self.norm_img.shape, strict=True)
        ):
            print(f"coords {coords} not in {img_layer}")
            return

        half = self.winsize // 2

        self.update_window_display(coords, half)

    def _on_run_bt(self):
        if self.norm_img is None:
            return
        if self.center_coords is None:
            return

        if self.model is None:
            print("Initialise model first")
            return

        half = self.winsize // 2

        factor = self.winsize // MODEL_WIN_SIZE

        patch = extract_tumor_window(self.norm_img, self.center_coords, half)

        if factor > 1:
            patch = downsize_window(patch, factor)
        assert all(s == MODEL_WIN_SIZE for s in patch.shape)

        patch_pred = single_image_prediction(
            self.model, patch, self.post_transform, self.device
        )

        if factor > 1:
            patch_pred = up_sample_pred(patch_pred, factor)
        assert all(s == self.winsize for s in patch_pred.shape)

        prediction = insert_patch(
            patch_pred,
            self.center_coords,
            self.prediction_layer.data.shape,
            half,
        )

        self.prediction_layer.data = prediction

    def _on_merge_bt(self):
        labels_shape = self.label_layer_cb.value.data.shape
        img_shape = self.img_layer_cb.value.data.shape
        if labels_shape != img_shape:
            raise RuntimeError(
                f"Shape mismatch: image layer has shape {img_shape} but labels layer has shape {labels_shape}. "
            )
        binary = self.prediction_layer.data.astype(bool)

        val = self.target_sb.value

        self.label_layer_cb.value.data[binary] = val
        self.label_layer_cb.value.refresh()
        self.target_sb.value += 1

    def _on_label_layer_changed(self, layer):
        if layer is None:
            return
        val = int(layer.data.max()) + 1
        self.target_sb.value = val

    def _on_close(self):
        if (
            self.window_display_layer is not None
            and self.window_display_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.window_display_layer)
        if (
            self.prediction_layer is not None
            and self.prediction_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.prediction_layer)

    def update_window_display(self, coords, half):
        z, y, x = coords

        if (
            self.window_display_layer is None
            or self.window_display_layer not in self.viewer.layers
        ):
            self._on_img_layer_changed(self.img_layer_cb.value)

        self.window_display_layer.data[:] = 0
        shape = self.window_display_layer.data.shape
        z0, z1 = max(0, z - half), min(shape[0], z + half)
        y0, y1 = max(0, y - half), min(shape[1], y + half)
        x0, x1 = max(0, x - half), min(shape[2], x + half)

        self.window_display_layer.data[z0:z1, y0:y1, x0:x1] = 1
        self.window_display_layer.refresh()

    def refresh_plugin_display_layers(self):
        if self.window_display_layer is not None:
            if self.window_display_layer in self.viewer.layers:
                self.viewer.layers.remove(self.window_display_layer)
            self.window_display_layer = None
        self.center_coords = None

        if self.prediction_layer is not None:
            if self.prediction_layer in self.viewer.layers:
                self.viewer.layers.remove(self.prediction_layer)
            self.prediction_layer = None

        img_layer = self.img_layer_cb.value

        if img_layer is not None:
            empty_data = np.zeros(img_layer.data.shape, dtype=np.uint8)
            self.window_display_layer = self.viewer.add_labels(
                empty_data,
                name="Window Display",
                opacity=0.4,
            )
            self.window_display_layer.colormap = (
                napari.utils.DirectLabelColormap(
                    color_dict={None: "transparent", 1: [0, 1, 0, 0.4]}
                )
            )

            empty_data = np.zeros(img_layer.data.shape, dtype=np.uint8)
            self.prediction_layer = self.viewer.add_labels(
                empty_data, name="Prediction", opacity=1
            )
            self.prediction_layer.contour = 1
            self.prediction_layer.colormap = napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", 1: [0, 0.8, 1, 1]}
            )
