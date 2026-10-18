package com.google.ar.core.examples.java.rawdepth;

import android.media.Image;
import android.util.Log;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.exceptions.NotYetAvailableException;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Appends text on one background thread. Reuse one instance for each recording session. */
public class DataSave {
    private static final String TAG = DataSave.class.getSimpleName();

    private final ThreadPoolExecutor diskWriter = new ThreadPoolExecutor(
            1, 1,
            0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<Runnable>(4),
            new ThreadPoolExecutor.AbortPolicy());

    /**
     * Queues a UTF-8 line without waiting for disk I/O. The parent directory must exist.
     *
     * @return true if queued, not necessarily saved; false if full or shut down.
     *     Write failures are reported to Logcat from the worker thread.
     */
    public boolean appendAsync(File file, String line) {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(line, "line");

        try {
            diskWriter.execute(() -> {
                try {
                    appendLine(file, line);
                } catch (IOException e) {
                    Log.e(TAG, "Capture write failed: " + file, e);
                }
            });
            return true;
        } catch (RejectedExecutionException e) {
            Log.w(TAG, "Writer unavailable or queue full; capture skipped");
            return false;
        }
    }

    /** Stops accepting work; queued writes finish in the background. Does not wait for them. */
    public void shutdown() {
        diskWriter.shutdown();
    }

    private static void appendLine(File file, String line) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(
                        new FileOutputStream(file, true),
                        StandardCharsets.UTF_8))) {

            writer.write(line);
            writer.newLine();
        }
    }

    /**
     * Serializes every valid point in the supplied depth frame as one JSONL record (without a
     * newline). DepthData.create retains all nonzero depth pixels; no sampling is applied here.
     * Each point is [x, y, z, confidence, r, g, b]: XYZ are in meters relative to the
     * frame's anchor; confidence and RGB are in [0, 1]. modelMatrix is the column-major
     * transform from those coordinates to session coordinates, captured at export time.
     */
    public static String GetData(DepthData data) {
        Objects.requireNonNull(data, "data");
        FloatBuffer points = data.getPoints();
        FloatBuffer colors = data.getColors();
        int pointCount = points.limit() / 4;
        if (points.limit() % 4 != 0 || colors.limit() != pointCount * 3) {
            throw new IllegalArgumentException("Point and RGB buffers must contain matching points");
        }

        float[] modelMatrix = new float[16];
        data.getModelMatrix(modelMatrix);
        StringBuilder result = new StringBuilder();
        result.append("{\"timestampNs\":").append(data.getTimestamp())
                .append(",\"modelMatrix\":[");
        for (int i = 0; i < modelMatrix.length; i++) {
            if (i > 0) result.append(',');
            result.append(modelMatrix[i]);
        }
        result.append("],\"points\":[");
        for (int i = 0; i < pointCount; i++) {
            if (i > 0) result.append(',');
            int p = i * 4;
            int c = i * 3;
            // Absolute reads preserve the buffer positions used by the renderer.
            result.append('[')
                    .append(points.get(p)).append(',')
                    .append(points.get(p + 1)).append(',')
                    .append(points.get(p + 2)).append(',')
                    .append(points.get(p + 3)).append(',')
                    .append(colors.get(c)).append(',')
                    .append(colors.get(c + 1)).append(',')
                    .append(colors.get(c + 2)).append(']');
        }
        return result.append("]}").toString();
    }

    /**
     * Serializes the current frame's full raw depth and confidence images as one JSONL record.
     * depthMillimeters and confidence are flat row-major arrays: index = y * width + x.
     * Depth is unsigned 16-bit millimeters (zero means missing); confidence is unsigned 0-255.
     * No pixels are subsampled or filtered. RGB camera imagery is not included.
     *
     * textureIntrinsics retains calibration at its original image dimensions. Scale fx/cx by
     * depth width / calibration width and fy/cy by depth height / calibration height to recover
     * depth-pixel intrinsics, as in PointCloudHelper. modelMatrix is the column-major transform
     * from camera-local coordinates (X right, Y up, Z backward) to session coordinates.
     *
     * Call while frame is current, before the next Session.update(). All acquired images are
     * closed before returning; only the resulting string should be passed to the disk writer.
     *
     * @throws NotYetAvailableException if depth or confidence is not available for this frame
     */
    public static String GetData(Frame frame) throws NotYetAvailableException {
        Objects.requireNonNull(frame, "frame");
        try (Image depth = frame.acquireRawDepthImage16Bits();
                Image confidence = frame.acquireRawDepthConfidenceImage()) {
            int width = depth.getWidth();
            int height = depth.getHeight();
            if (confidence.getWidth() != width || confidence.getHeight() != height) {
                throw new IllegalArgumentException("Depth and confidence image dimensions must match");
            }

            Image.Plane depthPlane = depth.getPlanes()[0];
            Image.Plane confidencePlane = confidence.getPlanes()[0];
            ByteBuffer depthBuffer = depthPlane.getBuffer().duplicate().order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer confidenceBuffer = confidencePlane.getBuffer().duplicate();
            int depthStart = depthBuffer.position();
            int confidenceStart = confidenceBuffer.position();
            int depthRowStride = depthPlane.getRowStride();
            int depthPixelStride = depthPlane.getPixelStride();
            int confidenceRowStride = confidencePlane.getRowStride();
            int confidencePixelStride = confidencePlane.getPixelStride();

            Camera camera = frame.getCamera();
            CameraIntrinsics intrinsics = camera.getTextureIntrinsics();
            int[] dimensions = intrinsics.getImageDimensions();
            float[] focalLength = intrinsics.getFocalLength();
            float[] principalPoint = intrinsics.getPrincipalPoint();
            float[] modelMatrix = new float[16];
            camera.getPose().toMatrix(modelMatrix, 0);

            StringBuilder result = new StringBuilder();
            result.append("{\"type\":\"rawDepth\",\"timestampNs\":").append(depth.getTimestamp())
                    .append(",\"frameTimestampNs\":").append(frame.getTimestamp())
                    .append(",\"width\":").append(width)
                    .append(",\"height\":").append(height)
                    .append(",\"textureIntrinsics\":{\"width\":").append(dimensions[0])
                    .append(",\"height\":").append(dimensions[1])
                    .append(",\"fx\":").append(focalLength[0])
                    .append(",\"fy\":").append(focalLength[1])
                    .append(",\"cx\":").append(principalPoint[0])
                    .append(",\"cy\":").append(principalPoint[1])
                    .append("},\"modelMatrix\":[");
            for (int i = 0; i < modelMatrix.length; i++) {
                if (i > 0) result.append(',');
                result.append(modelMatrix[i]);
            }
            result.append("],\"depthMillimeters\":[");
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (y != 0 || x != 0) result.append(',');
                    int offset = depthStart + y * depthRowStride + x * depthPixelStride;
                    result.append(depthBuffer.getShort(offset) & 0xffff);
                }
            }
            result.append("],\"confidence\":[");
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (y != 0 || x != 0) result.append(',');
                    int offset = confidenceStart + y * confidenceRowStride + x * confidencePixelStride;
                    result.append(confidenceBuffer.get(offset) & 0xff);
                }
            }
            return result.append("]}").toString();
        }
    }

}
