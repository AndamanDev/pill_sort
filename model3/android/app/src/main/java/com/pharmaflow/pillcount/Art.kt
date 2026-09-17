package com.pharmaflow.pillcount

import android.content.Context
import java.io.File

/**
 * Unpack the PNGs `ui/` draws with into the app's private storage.
 *
 * `ui/text.py` renders every Thai word and every Latin glyph from a PNG made once from
 * a real typeface, because `cv2.putText` has only the 1960s Hershey plotter faces. On a
 * PC it finds them in `assets/` next to the code. Inside the APK there is no "next to
 * the code" -- the Python is a zip entry -- so they ride along as Android assets and
 * land here, at a path handed to `ui/` through the PILLSORT_ASSETS environment variable.
 *
 * Unpacked once per INSTALL, and the difference matters. The stamp used to hold the
 * versionCode alone, which is derived from version.properties and therefore identical
 * across every build made between two releases -- so a phone that had the app already
 * kept the assets from the first install for ever. New words rendered as empty boxes and
 * a logo added later simply never appeared, while the APK on the desk plainly contained
 * both. The stamp now carries the install time as well, which Android changes whenever
 * the package is replaced, so sideloading a rebuild refreshes them the way a release
 * would.
 */
object Art {

    fun unpack(context: Context, mark: String): File {
        val dir = File(context.filesDir, "art")
        val stamp = File(dir, ".version")

        if (stamp.isFile && stamp.readText().trim() == mark) {
            return dir
        }

        // Not an incremental update. A rename in ui/ would otherwise leave the old file
        // behind for ever, and a stale glyph is the kind of thing that is noticed only
        // as "the 8 looks wrong".
        dir.deleteRecursively()
        dir.mkdirs()
        copyTree(context, "art", dir)
        stamp.writeText(mark)
        return dir
    }

    private fun copyTree(context: Context, assetPath: String, dest: File) {
        val names = context.assets.list(assetPath) ?: return
        if (names.isEmpty()) {
            // AssetManager.list cannot tell a file from an empty directory; reaching
            // here means it is a file.
            context.assets.open(assetPath).use { input ->
                dest.outputStream().use { input.copyTo(it) }
            }
            return
        }
        dest.mkdirs()
        for (name in names) {
            copyTree(context, "$assetPath/$name", File(dest, name))
        }
    }
}
