"""DC²Net dock: patch set, training, full-image inference, band importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project


class DeepLearningPanel(QWidget):
    """Workspace for the DC²Net branch."""

    message = Signal(str)
    artifacts_created = Signal()
    model_trained = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._cube: ArtifactRef | None = None
        self._patch_art: ArtifactRef | None = None
        self._model_art: ArtifactRef | None = None
        self._map_art: ArtifactRef | None = None
        self._importance_art: ArtifactRef | None = None
        self._last_path: Path | None = None

        root = QVBoxLayout(self)
        self.header = QLabel("未绑定立方体（在资源树中打开一个 CUBE）")
        self.header.setWordWrap(True)
        root.addWidget(self.header)
        root.addWidget(self._build_input_box())
        root.addWidget(self._build_train_box())

        actions = QHBoxLayout()
        self.btn_patch = QPushButton("构建图块集")
        self.btn_patch.clicked.connect(self._on_build_patchset)
        self.btn_train = QPushButton("训练 DC²Net")
        self.btn_train.clicked.connect(self._on_train)
        self.btn_infer = QPushButton("全图推理")
        self.btn_infer.clicked.connect(self._on_infer)
        self.btn_explain = QPushButton("波长重要性")
        self.btn_explain.clicked.connect(self._on_explain)
        for button in (self.btn_patch, self.btn_train, self.btn_infer, self.btn_explain):
            actions.addWidget(button)
        root.addLayout(actions)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(180)
        self.result.setPlaceholderText("训练日志、指标、推理与重要性结果会显示在这里。")
        root.addWidget(self.result, stretch=1)

        self.btn_open = QPushButton("打开最近产物")
        self.btn_open.clicked.connect(self._on_open_last)
        root.addWidget(self.btn_open)
        self._set_enabled(False)

    # ------------------------------------------------------------- building
    def _build_input_box(self) -> QGroupBox:
        box = QGroupBox("输入与图块")
        form = QFormLayout(box)
        self.mask_combo = QComboBox()
        self.mask_combo.setToolTip("用于切图的标签图（分割步骤的 *_labels 产物）")
        form.addRow("标签图", self.mask_combo)

        self.class_map_edit = QLineEdit()
        self.class_map_edit.setPlaceholderText("如 1-6:healthy, 7-12:diseased；留空则每个叶片一类")
        form.addRow("类别映射", self.class_map_edit)

        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(3, 63)
        self.patch_spin.setSingleStep(2)
        self.patch_spin.setValue(15)
        form.addRow("patch 大小", self.patch_spin)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 63)
        self.stride_spin.setValue(8)
        form.addRow("采样步长", self.stride_spin)
        return box

    def _build_train_box(self) -> QGroupBox:
        box = QGroupBox("训练与推理")
        form = QFormLayout(box)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(30)
        form.addRow("训练轮数", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        form.addRow("batch 大小", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-5, 1.0)
        self.lr_spin.setSingleStep(1e-3)
        self.lr_spin.setValue(3e-3)
        form.addRow("学习率", self.lr_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(8)
        form.addRow("早停耐心", self.patience_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(0)
        form.addRow("随机种子", self.seed_spin)

        self.device_combo = QComboBox()
        for label, value in (("自动", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")):
            self.device_combo.addItem(label, value)
        form.addRow("设备", self.device_combo)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(10)
        form.addRow("重要波段数", self.topk_spin)
        return box

    # -------------------------------------------------------------- binding
    def bind(self, project: Project, cube_art: ArtifactRef) -> None:
        self._project = project
        self._cube = cube_art
        self._patch_art = None
        self._model_art = None
        self._map_art = None
        self._importance_art = None
        self.header.setText(f"当前立方体：{cube_art.name}  ({cube_art.uri})")
        self._refresh_mask_combo()
        self._set_enabled(True)

    def unbind(self) -> None:
        self._project = None
        self._cube = None
        self._patch_art = None
        self._model_art = None
        self._map_art = None
        self._importance_art = None
        self.mask_combo.clear()
        self.header.setText("未绑定立方体（在资源树中打开一个 CUBE）")
        self._set_enabled(False)

    def _refresh_mask_combo(self) -> None:
        self.mask_combo.clear()
        if self._project is None:
            return
        for art in self._project.list_artifacts():
            if art.kind != ArtifactKind.MASK:
                continue
            if (art.metadata or {}).get("format") == "classification_map":
                continue
            self.mask_combo.addItem(art.name, art.id)
        if self.mask_combo.count() == 0:
            self.mask_combo.addItem("（暂无标签图，请先运行分割）", "")

    # --------------------------------------------------------------- params
    def _class_map(self) -> dict[str, str]:
        text = self.class_map_edit.text().strip()
        if not text:
            return {}
        mapping: dict[str, str] = {}
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"类别映射无法解析：{chunk!r}（应为 id:类别）")
            key, name = chunk.split(":", 1)
            key = key.strip()
            name = name.strip()
            if "-" in key:
                start, end = key.split("-", 1)
                for value in range(int(start), int(end) + 1):
                    mapping[str(value)] = name
            else:
                mapping[str(int(key))] = name
        return mapping

    def _run_step(
        self,
        name: str,
        inputs: list[ArtifactRef],
        params: dict[str, Any],
        seed: int | None = None,
    ) -> list[ArtifactRef]:
        assert self._project is not None
        kwargs: dict[str, Any] = {"params": params}
        if seed is not None:
            kwargs["seed"] = seed
        result = self._project.run_step(name, inputs, **kwargs)
        if not result.ok:
            raise RuntimeError(result.run.error or f"{name} 执行失败")
        outputs = result.outputs
        if result.skipped:
            outputs = self._project.find_cached_outputs(name, inputs, params=params, seed=seed)
        self.artifacts_created.emit()
        return outputs

    # ------------------------------------------------------------- actions
    def _on_build_patchset(self) -> None:
        if self._project is None or self._cube is None:
            return
        mask_id = str(self.mask_combo.currentData() or "")
        mask = self._project.get_artifact(mask_id) if mask_id else None
        if mask is None:
            QMessageBox.warning(self, "缺少标签图", "请先在分割面板生成标签图并选择它。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            params = {
                "patch_size": int(self.patch_spin.value()),
                "stride": int(self.stride_spin.value()),
                "class_map": self._class_map(),
            }
            seed = int(self.seed_spin.value())
            outputs = self._run_step("features.patchset", [self._cube, mask], params, seed)
            patch = next(
                (o for o in outputs if (o.metadata or {}).get("format") == "patchset"), None
            )
            if patch is None:
                raise RuntimeError("未产生 patchset 产物")
            self._patch_art = patch
            meta = patch.metadata or {}
            counts = meta.get("split_counts") or {}
            self.result.setPlainText(
                "\n".join(
                    [
                        f"图块集：{patch.name}",
                        f"样本数：{meta.get('n_samples')}  波段数：{meta.get('n_bands')}",
                        f"类别：{', '.join(meta.get('classes') or [])}",
                        f"划分：train={counts.get('train')} val={counts.get('val')} "
                        f"test={counts.get('test')}",
                        f"叶片分组：{meta.get('leaf_splits')}",
                        "",
                        "下一步：训练 DC²Net。",
                    ]
                )
            )
            self._last_path = patch.resolve(self._project.root)
            self._sync_actions()
            self.message.emit(f"图块集已生成：{patch.name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "构建图块集失败", str(exc))
            self.message.emit(f"构建图块集失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_train(self) -> None:
        if self._project is None or self._patch_art is None:
            QMessageBox.information(self, "训练", "请先构建图块集。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            params = {
                "epochs": int(self.epochs_spin.value()),
                "batch_size": int(self.batch_spin.value()),
                "lr": float(self.lr_spin.value()),
                "patience": int(self.patience_spin.value()),
                "device": str(self.device_combo.currentData()),
            }
            seed = int(self.seed_spin.value())
            outputs = self._run_step("dl.train", [self._patch_art], params, seed)
            model = next((o for o in outputs if o.kind == ArtifactKind.MODEL), None)
            if model is None:
                raise RuntimeError("训练未产生 MODEL 产物")
            self._model_art = model
            meta = model.metadata or {}
            metrics = meta.get("metrics") or {}
            lines = [
                f"模型：{model.name}",
                f"参数量：{meta.get('n_parameters'):,}  设备：{meta.get('device')}",
                f"最佳轮次 / 实际轮次：{meta.get('best_epoch')} / {meta.get('epochs_run')}"
                f"（{meta.get('stop_reason')}）",
                f"训练准确率：{_fmt(meta.get('train_accuracy'))}",
                "",
                f"测试 Accuracy : {_fmt(metrics.get('accuracy'))}",
                f"测试 F1(macro): {_fmt(metrics.get('f1_macro'))}",
                f"峰值显存/内存：{_fmt(meta.get('peak_memory_mb'))} MB",
                "",
                f"权重：{model.uri}",
            ]
            self.result.setPlainText("\n".join(lines))
            self._last_path = model.resolve(self._project.root)
            self._sync_actions()
            self.model_trained.emit(model)
            self.message.emit(f"DC²Net 训练完成：{model.name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "训练失败", str(exc))
            self.message.emit(f"训练失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_infer(self) -> None:
        if self._project is None or self._model_art is None or self._cube is None:
            QMessageBox.information(self, "推理", "请先训练模型。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            outputs = self._run_step("predict.map", [self._model_art, self._cube], {})
            mask = next((o for o in outputs if o.kind == ArtifactKind.MASK), None)
            if mask is None:
                raise RuntimeError("推理未产生分类图产物")
            self._map_art = mask
            meta = mask.metadata or {}
            counts = meta.get("class_counts") or {}
            lines = [
                f"分类图：{mask.name}  尺寸：{meta.get('shape')}",
                f"窗口 / 步长：{meta.get('patch_size')} / {meta.get('stride')}",
                f"拼接接缝比：{_fmt(meta.get('seam_ratio'))}（≈1 表示无缝）",
                "",
                "类别像素统计：",
            ]
            lines += [f"  {name}: {value}" for name, value in counts.items()]
            lines += ["", f"置信度图：{(meta or {}).get('confidence_uri')}"]
            self.result.setPlainText("\n".join(lines))
            self._last_path = mask.resolve(self._project.root)
            self.message.emit(f"全图推理完成：{mask.name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "推理失败", str(exc))
            self.message.emit(f"推理失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_explain(self) -> None:
        if self._project is None or self._model_art is None or self._patch_art is None:
            QMessageBox.information(self, "重要性", "请先训练模型。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            params = {"top_k": int(self.topk_spin.value())}
            outputs = self._run_step(
                "explain.wavelength", [self._model_art, self._patch_art], params
            )
            imp = next(
                (o for o in outputs if (o.metadata or {}).get("format") == "wavelength_importance"),
                None,
            )
            if imp is None:
                raise RuntimeError("未产生波长重要性产物")
            self._importance_art = imp
            meta = imp.metadata or {}
            lines = [f"重要性文件：{imp.uri}", "", "Top 波段："]
            for item in meta.get("top_bands") or []:
                wl = item.get("wavelength_nm")
                wl_text = f"{wl:.1f} nm" if isinstance(wl, (int, float)) else "—"
                lines.append(
                    f"  band {item.get('band_index')}: {wl_text}  "
                    f"importance={_fmt(item.get('importance'))}"
                )
            lines += ["", f"曲线图：{meta.get('svg_uri')}", f"波段列表：{meta.get('selected_uri')}"]
            self.result.setPlainText("\n".join(lines))
            self._last_path = imp.resolve(self._project.root)
            self.message.emit("波长重要性已导出")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "重要性分析失败", str(exc))
            self.message.emit(f"重要性分析失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_open_last(self) -> None:
        if self._last_path is None or not self._last_path.exists():
            return
        target = self._last_path
        if target.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    # ------------------------------------------------------------- helpers
    def _sync_actions(self) -> None:
        ready = self._project is not None and self._cube is not None
        self.btn_patch.setEnabled(ready)
        self.btn_train.setEnabled(ready and self._patch_art is not None)
        self.btn_infer.setEnabled(ready and self._model_art is not None)
        self.btn_explain.setEnabled(
            ready and self._model_art is not None and self._patch_art is not None
        )

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mask_combo,
            self.class_map_edit,
            self.patch_spin,
            self.stride_spin,
            self.epochs_spin,
            self.batch_spin,
            self.lr_spin,
            self.patience_spin,
            self.seed_spin,
            self.device_combo,
            self.topk_spin,
            self.btn_open,
        ):
            widget.setEnabled(enabled)
        self._sync_actions()


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v or v in (float("inf"), float("-inf")):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.{digits}f}"
