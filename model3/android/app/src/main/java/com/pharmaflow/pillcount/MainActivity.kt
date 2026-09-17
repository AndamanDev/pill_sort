package com.pharmaflow.pillcount

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.display.DisplayManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.Gravity
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Executors

/**
 * DrugCount's bench window, on a phone.
 *
 * What this class does NOT do is worth stating first, because it is most of the design:
 * it does not draw the count, the target, the buttons, the tray outlines or the status
 * pill. `ui/screen.py` composes all of that, exactly as it does on a PC, and this
 * activity's display is that image in an ImageView. A tap is mapped back into canvas
 * coordinates and handed to `ui/screen.py`'s own hit-testing, so the buttons are
 * wherever it drew them.
 *
 * The camera is the one exception, and it was not always. `compose()` used to paste the
 * video into that same image, which meant the preview could only refresh once the model
 * had finished with the frame: 1.3 fps on the test handset, of which 374 ms was the
 * forward pass. No amount of tuning fixes that -- four times faster is still 5 fps --
 * because the picture was chained to the pipeline at all. So the video is a CameraX
 * [PreviewView] now, composited by the display underneath a `compose(overlay = true)`
 * canvas that leaves the pane transparent. The camera runs at the display's rate. The
 * dots and the count arrive a frame or two behind it, which is what they are.
 *
 * The canvas is 1440x720 and landscape, which is the shape the bench window has and the
 * shape a tray is. Its size and the video pane inside it both come back from `start()`
 * rather than being written down on this side as well.
 *
 * The loop per frame:
 *
 *     CameraX (RGBA_8888)  ->  pillcount_android.frame()  ->  RGBA bytes  ->  Bitmap
 *
 * with the model's forward pass alone stepping back out to Kotlin, into [OrtEngine].
 * That analysis stream is [ANALYSIS] rather than anything larger: it is no longer what
 * anyone looks at, only what the detector reads, and it is letterboxed to 480x640 the
 * moment it reaches Python.
 */
class MainActivity : ComponentActivity() {

    private lateinit var canvasView: ImageView
    private lateinit var statusView: TextView
    private lateinit var previewView: PreviewView

    /** The splash: see [showLoading]. Every part of it is held, because every part moves. */
    private lateinit var loadingView: View
    private lateinit var loadingLogo: ImageView
    private lateinit var loadingTitle: TextView
    private lateinit var loadingBar: ProgressBar
    private lateinit var loadingPct: TextView
    private lateinit var loadingNote: TextView

    /** Set the first time a real frame reaches the screen, so the splash leaves once. */
    private var loadingDone = false

    /**
     * When the last camera frame ARRIVED, on the monotonic clock. Read by [checkCamera].
     *
     * Stamped on the way in rather than on the way out: a forward pass is most of a second
     * on this handset, and a stall measured from the end of the work would be declared on
     * every slow frame -- the app would announce a broken camera because its own model is
     * thinking. [analysing] covers the same hole from the other side.
     */
    @Volatile private var lastFrameAt = 0L

    /** True while a frame is inside the pipeline, so its own cost is not read as a stall. */
    @Volatile private var analysing = false

    /** When the camera was last rebound, so a dead camera is not rebound every tick. */
    private var lastRebindAt = 0L

    /** The display rotation the camera is currently aimed for. -1 until it is bound. */
    private var boundRotation = -1

    /** Kept so a turn of the phone can re-aim them without rebuilding the pipeline. */
    private var analysisUseCase: ImageAnalysis? = null
    private var previewUseCase: Preview? = null

    /**
     * TURNING THE PHONE END FOR END DOES NOT CHANGE ITS CONFIGURATION, and that is the
     * whole reason this listener exists.
     *
     * This activity is sensorLandscape, so both ways round are "landscape": flipping it
     * fires no onConfigurationChanged, recreates nothing, and rebinds nothing. What does
     * change is the display's rotation, from 90 to 270 -- and ImageAnalysis keeps the
     * target rotation it was built with, so the buffer it hands Python arrives UPSIDE
     * DOWN with respect to the preview the operator is looking at.
     *
     * That is not a cosmetic fault. The preview is CameraX's own surface and is always
     * the right way up; the dots are measured on the analysis image and drawn over that
     * preview, so every dot lands 180 degrees away from its pill. Measured on the test
     * handset at ROTATION_270: a pill the model placed at (0.675, 0.323) of the frame
     * appeared on screen at (0.327, 0.675), which is that point rotated by half a turn.
     */
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            val now = previewView.display?.rotation ?: return
            if (now == boundRotation) return
            Log.i(TAG, "display rotation $boundRotation -> $now, re-aiming the analyser")
            boundRotation = now
            // RE-AIMED, NOT REBOUND. Setting targetRotation is what CameraX asks for here:
            // it changes the rotationDegrees reported on the frames that follow and costs
            // nothing. Unbinding and binding again would do the same job and take the
            // camera down for a few hundred milliseconds to do it -- a black pane and a
            // dropped count every time somebody turns the phone round on the bench.
            analysisUseCase?.targetRotation = now
            previewUseCase?.targetRotation = now
        }
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * THE CAMERA CAN STOP AND NOTHING WOULD HAVE NOTICED, which is why this exists.
     *
     * The whole screen is composed by the analyzer, so when frames stop arriving the app
     * stops running: the last picture stays up, the count beside it stays up, and both
     * look current. The bench has a repaint timer of its own and catches this in
     * app/window.py; here, this is that timer. See pillcount_android.idle().
     */
    private val watchdog = object : Runnable {
        override fun run() {
            checkCamera()
            mainHandler.postDelayed(this, WATCH_MS)
        }
    }

    private val worker = Executors.newSingleThreadExecutor()

    /**
     * Touches run here, NOT on [worker].
     *
     * They shared a thread, and a thread that is busy with a forward pass is a thread
     * that is not listening: every press waited for whatever frame was in flight, which
     * on this handset is a few hundred milliseconds, and the whole app felt like it was
     * thinking about it. A press changes a number in Python and that is all it does --
     * it has no reason to queue behind the model.
     */
    private val touchWorker = Executors.newSingleThreadExecutor()

    private var bridge: PyObject? = null
    private var engine: OrtEngine? = null

    private var canvasW = 0
    private var canvasH = 0

    /** The video pane in canvas coordinates, straight from `ui.screen.pane()`. */
    private var pane = intArrayOf(0, 0, 0, 0, 0)

    /**
     * Two bitmaps, swapped. The worker thread fills one while the UI thread is drawing
     * the other; with a single bitmap a 4 MB pixel copy lands underneath a draw in
     * progress and the screen tears across a band of the image.
     */
    private var bitmaps: Array<Bitmap>? = null
    private var next = 0

    /** Reused between frames. A fresh 3.7 MB array per frame is pure GC pressure. */
    private var frameBuf = ByteArray(0)

    /** The geometry line is worth one entry in the log and not one per frame. */
    private var loggedGeometry = false

    /**
     * The bound camera, kept for its [androidx.camera.core.CameraControl].
     *
     * The bench window drives a UVC webcam's SHUTTER TIME through OpenCV; there is no
     * equivalent here. CameraX offers exposure COMPENSATION -- an index in device-defined
     * EV steps over a range the device reports -- which is a different control with a
     * different meaning, and that is exactly why `ui/screen.py` takes the slider's label
     * as text instead of working it out for itself.
     */
    private var camera: Camera? = null

    /**
     * Long press is this app's substitute for the bench's `r` key: a phone has no
     * keyboard, and the counting frame has to be started and saved somehow. A tap on the
     * video would do it by accident; a long press is the standard "edit this" gesture and
     * cannot be hit while reaching for a button.
     */
    private var gestures: GestureDetector? = null

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else status("ไม่ได้รับสิทธิ์กล้อง\nCamera permission denied")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        goFullscreen()
        canvasView = findViewById(R.id.canvas)
        statusView = findViewById(R.id.status)
        previewView = findViewById(R.id.preview)
        loadingView = findViewById(R.id.loading)
        loadingLogo = findViewById(R.id.loadingLogo)
        loadingTitle = findViewById(R.id.loadingTitle)
        loadingBar = findViewById(R.id.loadingBar)
        loadingPct = findViewById(R.id.loadingPct)
        loadingNote = findViewById(R.id.loadingNote)

        // FILL_CENTER is `ui.draw.cover`'s rule -- scale to fill, centre-crop the
        // overflow -- so the surface frames the scene the same way the pasted video
        // used to, and `screen.cover_map` can place the dots on it.
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // The overlay is what is touched; the surface below it must never take the tap.
        //
        // Every phase goes to Python, not just the release. The counting frame is dragged
        // and the brightness slider is dragged, and a drag reconstructed from its last
        // point is not a drag. The STATE MACHINE is on the Python side -- which corner is
        // held, where the rectangle is -- so the bench and the phone run one copy of it
        // and cannot disagree about what a drag means.
        gestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onLongPress(e: MotionEvent) {
                touch("long", canvasView, e)
            }
        })
        canvasView.setOnTouchListener { view, event ->
            gestures?.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> touch("down", view, event)
                MotionEvent.ACTION_MOVE -> touch("move", view, event)
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                    touch("up", view, event)
            }
            true
        }

        // The pane is in canvas coordinates, and the canvas is letterboxed into the
        // ImageView -- so where it lands on screen is only known once the view has been
        // measured, and again after any resize.
        canvasView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> placePreview() }

        showLoading()
        worker.execute { boot() }
    }

    /**
     * Everything that must happen before the first frame, on the worker thread.
     *
     * Chaquopy's first start on a new install extracts CPython's standard library,
     * numpy and OpenCV out of the APK -- seconds, not milliseconds -- and building the
     * ONNX Runtime session is another second on top. Doing it here rather than lazily
     * in the analyzer means the camera does not open onto a frozen screen.
     */
    private fun boot() {
        try {
            // The version AND when this copy was installed. Two builds of 0.1.0 are the
            // same version and not the same assets, and the phone has to notice.
            val pkg = packageManager.getPackageInfo(packageName, 0)
            @Suppress("DEPRECATION")
            val version = "${pkg.versionCode}-${pkg.lastUpdateTime}"
            val artDir = Art.unpack(this, version)
            step(20, "เตรียมไฟล์หน้าจอ")

            val modelBytes = assets.open("model/pillcount-det-v3.onnx").use { it.readBytes() }
            step(30, "วัดความเร็วโมเดล")
            // The preferences are where the provider race's winner is kept, under the
            // same install stamp the assets use: a new APK brings a new model and a new
            // ONNX Runtime, and last week's answer is about neither of them.
            val ort = OrtEngine(modelBytes,
                                getSharedPreferences(ENGINE_PREFS, MODE_PRIVATE), version)
            engine = ort
            step(45, "โหลดโมเดลตรวจจับเม็ดยา")

            if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
            val module = Python.getInstance().getModule("pillcount_android")
            step(70, "โหลดไลบรารีประมวลผลภาพ")
            val info = JSONObject(
                module.callAttr("start", artDir.absolutePath,
                    File(getExternalFilesDir(null), "records").absolutePath,
                    ort,
                    // THE VIEW'S size, not the display's. displayMetrics reports the
                    // screen, which is not the same rectangle: on a phone with a cutout
                    // or with the gesture bar still laid out, it is a few dozen pixels
                    // taller than the window the canvas is actually stretched into, and
                    // a canvas of the wrong shape is letterboxed with exactly the grey
                    // bands this was meant to remove. The view is measured by now --
                    // boot() runs after the first layout -- and falls back to the
                    // display only if it somehow is not.
                    canvasView.width.takeIf { it > 0 }
                        ?: resources.displayMetrics.widthPixels,
                    canvasView.height.takeIf { it > 0 }
                        ?: resources.displayMetrics.heightPixels).toString()
            )
            canvasW = info.getInt("width")
            canvasH = info.getInt("height")
            val paneJson = info.getJSONArray("pane")
            pane = IntArray(paneJson.length()) { paneJson.getInt(it) }
            bitmaps = Array(2) { Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888) }
            bridge = module
            step(88, "เตรียมการนับ")

            Log.i(TAG, "ready: $info")
            runOnUiThread {
                step(95, "เชื่อมต่อกล้อง")
                mainHandler.postDelayed(watchdog, WATCH_MS)
                placePreview()
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
                ) startCamera() else askCamera.launch(Manifest.permission.CAMERA)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "startup failed", t)
            status("เริ่มระบบไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                // THE ROTATION THE SCREEN IS AT RIGHT NOW, asked once and given to both
                // use cases. Left unset, ImageAnalysis keeps whatever the display was
                // doing when it was constructed, which on a phone that has been turned
                // since is a buffer half a turn away from the picture. See
                // [displayListener].
                val rotation = previewView.display?.rotation ?: Surface.ROTATION_0

                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(rotation)
                    // RGBA rather than the YUV_420_888 default. The conversion is
                    // CameraX's, done in native code on the way out, and it saves
                    // reassembling three planes with their own strides in Python for
                    // every frame.
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    // Small, now that this stream is not also the picture on screen.
                    // The model letterboxes whatever it gets to 448x448 and the colour
                    // measurements read a few hundred pixels per pill, so 720p was
                    // being paid for in cvtColor, rotate and the BGR->LAB conversion --
                    // 56 + 40 ms a frame -- and then thrown away.
                    .setTargetResolution(ANALYSIS)
                    // The pipeline is slower than the camera. Keeping the latest frame
                    // and dropping the rest is what makes the screen show now rather
                    // than a queue of the recent past.
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(worker, ::analyze)

                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                // A ViewPort, and it is load-bearing rather than tidy.
                //
                // The dots are measured on the analysis stream and drawn over the
                // preview surface, so the two have to be showing the same thing. The
                // first attempt simply asked both for 16:9 and assumed that settled it.
                // It does not: setTargetResolution is best-effort, the camera is free
                // to answer a 4:3 stream, and the analyzer then measures a wider field
                // of view than the preview displays -- which puts every dot off its
                // pill by a margin that grows towards the edges of the frame.
                //
                // A ViewPort states the requirement instead of hoping for it. CameraX
                // gives every use case in the group a crop rectangle covering the same
                // region, whatever resolutions it picked, and hands it over on each
                // frame as ImageProxy.cropRect. Crop to that and the analysis image IS
                // what the surface is showing, at the pane's own aspect ratio.
                val viewPort = ViewPort.Builder(Rational(pane[2], pane[3]), rotation)
                    .setScaleType(ViewPort.FILL_CENTER)   // PreviewView's scale type
                    .build()

                analysisUseCase = analysis
                previewUseCase = preview

                val group = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()

                val bound = future.get().let {
                    it.unbindAll()
                    it.bindToLifecycle(this@MainActivity,
                                       CameraSelector.DEFAULT_BACK_CAMERA, group)
                }
                camera = bound
                boundRotation = rotation

                // There WAS an exposure slider here, and the plumbing for it outlived the
                // control. The phone's screen was rebuilt to match the bench's, which has
                // no such slider -- the bench sets a webcam's shutter once at startup and
                // leaves it -- so `set_exposure_range` went out of the Python side and
                // this call stayed behind, throwing AttributeError into the log on every
                // single launch and telling nobody anything. Dead code that fails loudly
                // is still dead code.

                runOnUiThread {
                    statusView.visibility = View.GONE
                    placePreview()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "camera failed", t)
                status("เปิดกล้องไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val py = bridge
        val bufs = bitmaps
        if (py == null || bufs == null) {
            image.close()
            return
        }
        lastFrameAt = SystemClock.elapsedRealtime()
        analysing = true
        try {
            val plane = image.planes[0]
            val src = plane.buffer
            if (frameBuf.size != src.remaining()) frameBuf = ByteArray(src.remaining())
            src.get(frameBuf)

            val crop = image.cropRect
            if (!loggedGeometry) {
                loggedGeometry = true
                Log.i(TAG, "analysis ${image.width}x${image.height}" +
                        " crop=${crop.left},${crop.top},${crop.right},${crop.bottom}" +
                        " (${crop.width()}x${crop.height()})" +
                        " rotation=${image.imageInfo.rotationDegrees}" +
                        " pane=${pane[2]}x${pane[3]}")
            }

            val rgba = py.callAttr(
                "frame", frameBuf, image.width, image.height,
                plane.rowStride, image.imageInfo.rotationDegrees,
                crop.left, crop.top, crop.right, crop.bottom
            ).toJava(ByteArray::class.java)

            // An empty array means Python decided the screen is identical to the one
            // already on it. Skipping the upload is most of what makes the app feel
            // quick: the bitmap copy and the view invalidation are the expensive half of
            // a frame that would have shown exactly the same pixels.
            //
            // A plain if rather than an early return: this is inside `analyze`, whose
            // finally clause closes the ImageProxy, and jumping out of it past that is
            // how a camera pipeline stalls on its fourth frame.
            show(rgba)
        } catch (t: Throwable) {
            Log.e(TAG, "frame failed", t)
            status("ประมวลผลภาพไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
        } finally {
            analysing = false
            image.close()
        }
    }

    /**
     * Put a composed screen up. Called from the analyzer and from the watchdog alike.
     *
     * An EMPTY array means Python decided the screen is identical to the one already on
     * it. Skipping the upload is most of what makes the app feel quick: the bitmap copy
     * and the view invalidation are the expensive half of a frame that would have shown
     * exactly the same pixels.
     *
     * Both callers run on [worker], which is a single thread -- so `next` is not shared
     * across threads and the two bitmaps cannot be swapped out from under each other.
     */
    private fun show(rgba: ByteArray) {
        val bufs = bitmaps ?: return
        if (rgba.isEmpty()) return
        val bmp = bufs[next]
        next = 1 - next
        bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
        runOnUiThread {
            canvasView.setImageBitmap(bmp)
            if (!loadingDone) {
                // THE SPLASH LEAVES WHEN THERE IS SOMETHING BEHIND IT, not when boot()
                // returns. Binding the camera and waiting for its first frame is another
                // moment on top of everything boot() did, and hiding at the end of boot
                // left a bare background on screen for most of a second, which is exactly
                // the "has it crashed?" this screen exists to answer. The full bar stays
                // up for a beat so the last step is something the eye catches.
                loadingDone = true
                step(100, "พร้อมใช้งาน")
                canvasView.postDelayed({ hideLoading() }, 250)
            }
        }
    }

    /**
     * Has the camera gone quiet? Then redraw saying so, and eventually ask for it again.
     *
     * TWO SEPARATE JOBS, on two clocks. The redraw is what stops a stale count being
     * saved: Python works out the age of the picture, blocks the save button and puts a
     * red border round the video, exactly as the bench does. The rebind is the phone's
     * version of app/worker.py's `_reopen` -- unbinding and binding the use cases again
     * is what pulling the USB lead out and putting it back does by hand, and it is the
     * fix for a camera another app took, a device that was re-enumerated, or a driver
     * that dropped the session. It waits [REBIND_MS] because rebinding is not free and a
     * camera that is merely slow for a moment recovers without it.
     */
    private fun checkCamera() {
        val py = bridge ?: return
        if (lastFrameAt == 0L || analysing) return      // never started, or mid-frame
        val gap = SystemClock.elapsedRealtime() - lastFrameAt
        if (gap < STALL_MS) return

        worker.execute {
            try {
                show(py.callAttr("idle").toJava(ByteArray::class.java))
            } catch (t: Throwable) {
                Log.e(TAG, "idle redraw failed", t)
            }
        }

        val now = SystemClock.elapsedRealtime()
        if (gap > REBIND_MS && now - lastRebindAt > REBIND_MS) {
            lastRebindAt = now
            Log.w(TAG, "no camera frames for $gap ms, rebinding")
            startCamera()
        }
    }

    /**
     * Put the camera surface exactly where the overlay leaves a hole.
     *
     * `screen.pane()` reports the video pane in canvas coordinates; the canvas is drawn
     * fitCenter inside [canvasView], so the same scale and letterbox offsets the tap
     * handler undoes are applied forwards here. Both read one rectangle from Python,
     * which is what stops the hole and the surface from drifting apart when the panel
     * is resized.
     *
     * The corner radius is not applied to the surface. It does not need to be: the
     * overlay stays opaque over the corners -- `compose` punches a ROUNDED hole -- so
     * the card's corners are painted on top of the video, exactly as `rounded_paste`
     * used to repaint them.
     */
    private fun placePreview() {
        if (canvasW == 0 || pane[2] == 0) return
        val vw = canvasView.width
        val vh = canvasView.height
        if (vw == 0 || vh == 0) return

        val scale = minOf(vw.toFloat() / canvasW, vh.toFloat() / canvasH)
        val offX = (vw - canvasW * scale) / 2f
        val offY = (vh - canvasH * scale) / 2f

        val lp = previewView.layoutParams as FrameLayout.LayoutParams
        lp.width = (pane[2] * scale).toInt()
        lp.height = (pane[3] * scale).toInt()
        lp.leftMargin = (offX + pane[0] * scale).toInt()
        lp.topMargin = (offY + pane[1] * scale).toInt()
        lp.gravity = Gravity.TOP or Gravity.START
        previewView.layoutParams = lp
        previewView.visibility = View.VISIBLE
    }

    /**
     * One touch, in canvas coordinates, handed to Python.
     *
     * Undoing the letterbox here -- rather than asking the Python side to lay itself out
     * for this screen -- is what keeps one set of hit boxes: `hitboxes()` measures from
     * the same numbers `_chrome()` drew with, and neither knows a phone exists.
     *
     * The canvas is drawn fitCenter inside [canvasView], so the scale and letterbox this
     * undoes are the same ones [placePreview] applies forwards -- which is what keeps a
     * finger, a button and the video surface agreeing about where things are.
     *
     * Runs on the worker, never the UI thread: `touch` can write the counting frame to
     * disk, and a file write on the main thread is a frame dropped at best.
     */
    private fun touch(phase: String, view: View, event: MotionEvent) {
        val py = bridge ?: return
        if (canvasW == 0) return
        val scale = minOf(view.width.toFloat() / canvasW, view.height.toFloat() / canvasH)
        val x = (event.x - (view.width - canvasW * scale) / 2f) / scale
        val y = (event.y - (view.height - canvasH * scale) / 2f) / scale
        if (x < 0 || y < 0 || x > canvasW || y > canvasH) return
        touchWorker.execute {
            try {
                val reply = py.callAttr("touch", phase, x.toInt(), y.toInt()).toString()
                applyTouchReply(reply)
            } catch (t: Throwable) {
                Log.e(TAG, "touch failed", t)
            }
        }
    }

    /**
     * Act on whatever the touch changed. Python decides; this only carries it out.
     *
     * THERE IS NOTHING LEFT TO CARRY OUT, which is worth saying rather than deleting: the
     * target, the counting frame, the page and the records all live on the Python side
     * and are drawn from there on the next frame, so a press changes a number over there
     * and Kotlin has no part in it. The one exception used to be the exposure slider,
     * which the screen no longer has. Kept as the place a future one would go.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun applyTouchReply(reply: String) = Unit

    /**
     * The whole screen, with no system bars over it.
     *
     * The status bar sat across the top as an empty strip: this window has its own header
     * and no use for a second one, and on a bench the phone is a counter rather than a
     * phone. STICKY_IMMERSIVE rather than plain fullscreen so a swipe from the edge still
     * brings the bars back for a moment -- somebody has to be able to get out.
     */
    /**
     * The logo and a word, while Chaquopy unpacks CPython, numpy and OpenCV.
     *
     * That takes seconds on a new install and about one on every launch after it, and
     * until it finishes there is nothing to draw: the Python that draws this app's screen
     * is the thing being unpacked. An empty grey rectangle for a second reads as a crash,
     * so the same wordmark the header carries goes up first, from the APK's assets --
     * which Kotlin can read directly, without waiting for Art.unpack.
     */
    private fun showLoading() {
        try {
            // Read from assets, not res/: it is the same file the header draws, staged
            // out of model3/phone/assets by the build, so the wordmark on the splash and
            // the one in the app cannot come from two different pictures.
            assets.open("art/logo.png").use { input ->
                loadingLogo.setImageBitmap(android.graphics.BitmapFactory.decodeStream(input))
            }
        } catch (t: Throwable) {
            Log.w(TAG, "no logo for the loading screen", t)
            loadingLogo.visibility = View.GONE
        }
        loadingView.visibility = View.VISIBLE
        step(3, "กำลังเริ่มระบบ")
    }

    /**
     * One step of the start-up finished.
     *
     * THE PERCENTAGE IS NOT DECORATIVE, which is the note app/splash.py carries too: each
     * number is published when that step actually completes, so a bar that sits at 45%
     * says the ONNX session is the slow part, and one that sits at 20% says Chaquopy is
     * still unpacking numpy onto a new install. A spinner would say only that the phone
     * has not given up yet.
     */
    private fun step(percent: Int, what: String) = runOnUiThread {
        loadingBar.progress = percent
        loadingPct.text = "$percent%"
        loadingNote.text = what
    }

    /** Back to the real canvas: the frames that follow are the app, not a picture of it. */
    private fun hideLoading() = runOnUiThread {
        loadingView.visibility = View.GONE
        statusView.visibility = View.GONE
    }

    private fun goFullscreen() {
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN
                or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) goFullscreen()
    }

    private fun status(text: String) = runOnUiThread {
        if (loadingView.visibility == View.VISIBLE) {
            // app/splash.py's fail(), in Kotlin. The bar and the percentage mean "this is
            // still going", and once it is not they are a lie: a progress bar stopped at
            // 45% under a crash message is worse than no bar at all. The message takes the
            // step's place, in red, and the title says plainly that it did not open.
            loadingBar.visibility = View.GONE
            loadingPct.visibility = View.GONE
            loadingTitle.setTextColor(DANGER)
            loadingTitle.text = "เปิดโปรแกรมไม่สำเร็จ"
            loadingNote.setTextColor(DANGER)
            loadingNote.text = text.replace("\n", "   ")
        } else {
            statusView.text = text
            statusView.visibility = View.VISIBLE
        }
    }

    /**
     * Away: stop watching. CameraX unbinds with the lifecycle, so of course no frames are
     * arriving -- and a watchdog left running would sit there redrawing a red border onto
     * a screen nobody is looking at, once a second, for as long as the app is in the
     * background.
     */
    override fun onPause() {
        super.onPause()
        mainHandler.removeCallbacks(watchdog)
        try {
            (getSystemService(DISPLAY_SERVICE) as DisplayManager)
                .unregisterDisplayListener(displayListener)
        } catch (t: Throwable) {
            Log.w(TAG, "could not unregister the display listener", t)
        }
    }

    /**
     * The other half of the same problem: a turn that DOES change the configuration.
     *
     * Landscape to reverse landscape does not; anything through portrait does, and
     * configChanges says this activity handles it rather than being recreated. Either way
     * the answer is the same one -- bind the camera again for the rotation the screen is
     * at now -- so both routes lead here.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // A turn that changes the configuration changes the display's rotation too, and
        // the listener above deals with that. What is left here is the window: the canvas
        // is re-measured and the preview surface has to be put back under its hole.
        placePreview()
    }

    /**
     * Back: watch again, from NOW.
     *
     * The clock has to be reset as well as the callback re-posted. Without it the first
     * tick reads a gap the size of however long the phone was in a pocket and announces a
     * camera fault that was never a fault, a second before the rebound camera delivers
     * its first frame and clears it again.
     */
    override fun onResume() {
        super.onResume()
        if (bridge != null) {
            lastFrameAt = SystemClock.elapsedRealtime()
            mainHandler.removeCallbacks(watchdog)
            mainHandler.postDelayed(watchdog, WATCH_MS)
        }
        (getSystemService(DISPLAY_SERVICE) as DisplayManager)
            .registerDisplayListener(displayListener, mainHandler)
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(watchdog)
        touchWorker.shutdown()
        super.onDestroy()
        worker.shutdown()
        engine?.close()
    }

    private companion object {

        /** theme.py's DANGER, for the one thing on this side that has to be red. */
        val DANGER = 0xFFC0392B.toInt()

        /** Where OrtEngine remembers which execution provider won its race. */
        const val ENGINE_PREFS = "engine"

        /** How often the watchdog looks. Cheap: a subtraction, unless something is wrong. */
        const val WATCH_MS = 500L

        /**
         * Silence this long means the picture is not live any more.
         *
         * Under app/window.py's 1.5s, deliberately: Python is the one that decides what
         * counts as stale, and it should be asked a little before its own threshold so
         * the red border appears at 1.5s rather than at 1.5s plus a tick.
         */
        const val STALL_MS = 1000L

        /** How long a dead camera is left alone before the use cases are bound again. */
        const val REBIND_MS = 6000L
        const val TAG = "pillsort"

        /**
         * What the analyzer asks for. 16:9 to match the preview's aspect ratio -- the
         * dots are placed from this stream and drawn over that surface, and two
         * different shapes would be two different fields of view.
         */
        // 4:3, to match the pane the preview fills and the ViewPort built from it. At
        // 16:9 the ViewPort cropped a 480x360 strip out of the middle for the model to
        // read, which is both smaller than it needs and a different field of view from
        // the one the operator frames the tray in.
        val ANALYSIS = Size(640, 480)
    }
}
