import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Improved NMS function with model source tracking
def run_nms(yolo_boxes, yolo_scores, mask_boxes, mask_scores, iou_thresh=0.5):
    """Apply Non-Maximum Suppression (NMS) on detections from YOLOv5 and Mask R-CNN."""
    # Combine YOLOv5 and Mask R-CNN results
    combined_boxes = np.vstack((yolo_boxes, mask_boxes))
    combined_scores = np.concatenate((yolo_scores, mask_scores))
    model_sources = ['YOLOv5'] * len(yolo_scores) + ['Mask R-CNN'] * len(mask_scores)
    
    # Get coordinates of the boxes
    x1 = combined_boxes[:, 0]
    y1 = combined_boxes[:, 1]
    x2 = combined_boxes[:, 2]
    y2 = combined_boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    order = combined_scores.argsort()[::-1]  # Sort scores in descending order
    keep = []
    kept_sources = []
    
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        kept_sources.append(model_sources[idx])
        
        # Calculate intersection area
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        
        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)
        
        intersection = w * h
        union = areas[idx] + areas[order[1:]] - intersection
        
        # Calculate IoU (Intersection over Union)
        iou = intersection / union
        
        # Keep only boxes with IOU < threshold
        order = order[1:][iou <= iou_thresh]
    
    return keep, kept_sources

# App Config
st.set_page_config(page_title="Solar Filament Detection", layout="centered")
st.title("🌞 Solar Filament Detection using YOLOv5 and Mask R-CNN")

st.markdown(
    """
    Welcome to the **Solar Filament Detection App**.  
    Upload a solar image to detect filaments using YOLOv5 and Mask R-CNN models.
    """
)

# Sidebar
st.sidebar.header("🔍 Model Configuration")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IOU Threshold for NMS", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader("📤 Upload a solar image", type=['jpg', 'jpeg', 'png'])

# Initialize session state variables
if 'yolo_results' not in st.session_state:
    st.session_state.yolo_results = None
if 'mask_results' not in st.session_state:
    st.session_state.mask_results = None
if 'combined_results' not in st.session_state:
    st.session_state.combined_results = None

# Section 1: Upload Solar Image and Perform Detection
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_path = Path("uploaded_image.jpg")
    image.save(image_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_path, caption='🖼️ Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("### Model Selection")
        run_yolo = st.button("Run YOLOv5 Detection")
        run_mask = st.button("Run Mask R-CNN Detection")
        run_both = st.button("Run Both Models", type="primary")
    
    if run_yolo or run_both:
        with st.spinner("Running YOLOv5 detection..."):
            model = torch.hub.load(
                '/content/drive/MyDrive/SolarFilamentDetection/yolov5',
                'custom',
                path='/content/drive/MyDrive/SolarFilamentDetection/yolov5/runs/train/exp3/weights/last.pt',
                source='local',
                force_reload=True
            )
            model.conf = conf_thres
            results = model(str(image_path))
            results.render()
            st.session_state.yolo_results = {
                'image': Image.fromarray(results.ims[0]),
                'boxes': results.xyxy[0].cpu().numpy(),
                'scores': results.xyxy[0][:, 4].cpu().numpy()
            }
    
    if run_mask or run_both:
        with st.spinner("Running Mask R-CNN detection..."):
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/SolarFilamentDetection/output/model_final.pth"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            predictor = DefaultPredictor(cfg)
            img_cv2 = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            outputs = predictor(img_rgb)
            instances = outputs["instances"].to("cpu")
            
            if instances.has("pred_boxes") and len(instances.pred_boxes) > 0:
                mask_overlay = img_rgb.copy()
                for mask in instances.pred_masks:
                    mask = mask.numpy()
                    mask_overlay[mask] = [0, 255, 0]
                
                st.session_state.mask_results = {
                    'image': mask_overlay,
                    'boxes': instances.pred_boxes.tensor.numpy(),
                    'scores': instances.scores.numpy()
                }
            else:
                st.session_state.mask_results = None
                st.warning("No filaments detected by Mask R-CNN")

    # Display results and download options
    if st.session_state.yolo_results is not None:
        st.subheader("YOLOv5 Results")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(st.session_state.yolo_results['image'], 
                    caption=f'YOLOv5 Detection (Confidence ≥ {conf_thres})', 
                    use_column_width=True)
        
        with col2:
            # Download button for YOLOv5 results
            yolo_result_path = "yolo_result.jpg"
            st.session_state.yolo_results['image'].save(yolo_result_path)
            with open(yolo_result_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download YOLOv5 Results",
                    data=file,
                    file_name="yolo_detection.jpg",
                    mime="image/jpeg",
                    key="yolo_download"
                )
        
        with st.expander("YOLOv5 Detection Details"):
            st.write(f"Detected {len(st.session_state.yolo_results['boxes'])} filaments")
            yolo_df = pd.DataFrame({
                'x1': st.session_state.yolo_results['boxes'][:, 0],
                'y1': st.session_state.yolo_results['boxes'][:, 1],
                'x2': st.session_state.yolo_results['boxes'][:, 2],
                'y2': st.session_state.yolo_results['boxes'][:, 3],
                'Confidence': st.session_state.yolo_results['scores']
            })
            st.dataframe(yolo_df)
            
            # Download button for YOLOv5 data
            yolo_csv = yolo_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download YOLOv5 Detection Data (CSV)",
                data=yolo_csv,
                file_name="yolo_detections.csv",
                mime="text/csv",
                key="yolo_csv"
            )
    
    if st.session_state.mask_results is not None:
        st.subheader("Mask R-CNN Results")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(st.session_state.mask_results['image'], 
                    caption=f'Mask R-CNN Detection (Confidence ≥ {conf_thres})', 
                    use_column_width=True)
        
        with col2:
            # Download button for Mask R-CNN results
            mask_result_path = "mask_result.jpg"
            cv2.imwrite(mask_result_path, cv2.cvtColor(st.session_state.mask_results['image'], cv2.COLOR_RGB2BGR))
            with open(mask_result_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Mask R-CNN Results",
                    data=file,
                    file_name="mask_rcnn_detection.jpg",
                    mime="image/jpeg",
                    key="mask_download"
                )
        
        with st.expander("Mask R-CNN Detection Details"):
            st.write(f"Detected {len(st.session_state.mask_results['boxes'])} filaments")
            mask_df = pd.DataFrame({
                'x1': st.session_state.mask_results['boxes'][:, 0],
                'y1': st.session_state.mask_results['boxes'][:, 1],
                'x2': st.session_state.mask_results['boxes'][:, 2],
                'y2': st.session_state.mask_results['boxes'][:, 3],
                'Confidence': st.session_state.mask_results['scores']
            })
            st.dataframe(mask_df)
            
            # Download button for Mask R-CNN data
            mask_csv = mask_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Mask R-CNN Detection Data (CSV)",
                data=mask_csv,
                file_name="mask_detections.csv",
                mime="text/csv",
                key="mask_csv"
            )

    # Section 2: NMS on Detection Results
    if st.session_state.yolo_results is not None and st.session_state.mask_results is not None:
        st.subheader("🔄 Combined Detection with Non-Maximum Suppression")
        
        keep_indices, kept_sources = run_nms(
            st.session_state.yolo_results['boxes'][:, :4],
            st.session_state.yolo_results['scores'],
            st.session_state.mask_results['boxes'],
            st.session_state.mask_results['scores'],
            iou_thresh=iou_thres
        )
        
        # Combine all boxes and scores
        all_boxes = np.vstack((
            st.session_state.yolo_results['boxes'][:, :4],
            st.session_state.mask_results['boxes']
        ))
        all_scores = np.concatenate((
            st.session_state.yolo_results['scores'],
            st.session_state.mask_results['scores']
        ))
        
        # Get the kept boxes and their info
        nms_boxes = all_boxes[keep_indices]
        nms_scores = all_scores[keep_indices]
        
        # Create visualization
        combined_img = np.array(image.copy())
        
        # Draw boxes with different colors based on source
        for box, score, source in zip(nms_boxes, nms_scores, kept_sources):
            color = (255, 0, 0) if source == 'YOLOv5' else (0, 255, 0)
            cv2.rectangle(combined_img, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            label = f"{source}: {score:.2f}"
            cv2.putText(combined_img, label, 
                       (int(box[0]), int(box[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display combined results
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(combined_img, caption="Combined Detection After NMS", use_column_width=True)
        
        with col2:
            # Download button for combined results
            combined_result_path = "combined_result.jpg"
            cv2.imwrite(combined_result_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
            with open(combined_result_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Combined Results",
                    data=file,
                    file_name="combined_detection.jpg",
                    mime="image/jpeg",
                    key="combined_download"
                )
        
        # Show results table
        with st.expander("NMS Results Details"):
            st.write(f"Kept {len(nms_boxes)} detections after NMS (IOU threshold: {iou_thres})")
            results_df = pd.DataFrame({
                'x1': nms_boxes[:, 0],
                'y1': nms_boxes[:, 1],
                'x2': nms_boxes[:, 2],
                'y2': nms_boxes[:, 3],
                'Confidence': nms_scores,
                'Model': kept_sources
            })
            st.dataframe(results_df)
            
            # Download button for combined data
            combined_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Combined Detection Data (CSV)",
                data=combined_csv,
                file_name="combined_detections.csv",
                mime="text/csv",
                key="combined_csv"
            )
        
        # Save combined results to session state
        st.session_state.combined_results = {
            'image': combined_img,
            'boxes': nms_boxes,
            'scores': nms_scores,
            'sources': kept_sources
        }
    elif st.session_state.yolo_results is not None or st.session_state.mask_results is not None:
        st.warning("Please run both models to perform NMS combination")

# Add a section to download all results at once
if uploaded_file is not None and (st.session_state.yolo_results is not None or 
                                st.session_state.mask_results is not None or 
                                st.session_state.combined_results is not None):
    st.subheader("📦 Download All Results")
    
    # Create a zip file with all results
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add original image
        zip_file.write(image_path, "original_image.jpg")
        
        # Add YOLOv5 results if available
        if st.session_state.yolo_results is not None:
            yolo_result_path = "yolo_result.jpg"
            st.session_state.yolo_results['image'].save(yolo_result_path)
            zip_file.write(yolo_result_path, "yolo_detection.jpg")
            
            yolo_df = pd.DataFrame({
                'x1': st.session_state.yolo_results['boxes'][:, 0],
                'y1': st.session_state.yolo_results['boxes'][:, 1],
                'x2': st.session_state.yolo_results['boxes'][:, 2],
                'y2': st.session_state.yolo_results['boxes'][:, 3],
                'Confidence': st.session_state.yolo_results['scores']
            })
            zip_file.writestr("yolo_detections.csv", yolo_df.to_csv(index=False))
        
        # Add Mask R-CNN results if available
        if st.session_state.mask_results is not None:
            mask_result_path = "mask_result.jpg"
            cv2.imwrite(mask_result_path, cv2.cvtColor(st.session_state.mask_results['image'], cv2.COLOR_RGB2BGR))
            zip_file.write(mask_result_path, "mask_rcnn_detection.jpg")
            
            mask_df = pd.DataFrame({
                'x1': st.session_state.mask_results['boxes'][:, 0],
                'y1': st.session_state.mask_results['boxes'][:, 1],
                'x2': st.session_state.mask_results['boxes'][:, 2],
                'y2': st.session_state.mask_results['boxes'][:, 3],
                'Confidence': st.session_state.mask_results['scores']
            })
            zip_file.writestr("mask_detections.csv", mask_df.to_csv(index=False))
        
        # Add combined results if available
        if st.session_state.combined_results is not None:
            combined_result_path = "combined_result.jpg"
            cv2.imwrite(combined_result_path, cv2.cvtColor(st.session_state.combined_results['image'], cv2.COLOR_RGB2BGR))
            zip_file.write(combined_result_path, "combined_detection.jpg")
            
            combined_df = pd.DataFrame({
                'x1': st.session_state.combined_results['boxes'][:, 0],
                'y1': st.session_state.combined_results['boxes'][:, 1],
                'x2': st.session_state.combined_results['boxes'][:, 2],
                'y2': st.session_state.combined_results['boxes'][:, 3],
                'Confidence': st.session_state.combined_results['scores'],
                'Model': st.session_state.combined_results['sources']
            })
            zip_file.writestr("combined_detections.csv", combined_df.to_csv(index=False))
    
    # Download all button
    st.download_button(
        label="⬇️ Download All Results (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="solar_filament_detection_results.zip",
        mime="application/zip",
        key="all_results"
    )