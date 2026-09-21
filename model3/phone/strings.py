"""Every word the phone screen can show, in one list, because each one is a PNG.

THE PHONE CANNOT RENDER TEXT. Chaquopy gives numpy and OpenCV 4.5.1 and nothing else --
no Pillow, no font stack -- and cv2.putText knows only the Hershey plotter faces from the
1960s, which have no Thai at all. So every string here is drawn on the PC by
model3/tools/render_text.py, from the same IBM Plex Sans Thai the desktop window uses, and
shipped in the APK as an image. Thai has to be rendered as whole WORDS rather than letters
because its marks stack and reorder around the consonant they belong to; blitting
character by character would take ทึ apart and put the mark somewhere to the right of it.

WHICH IS WHY THIS FILE EXISTS RATHER THAN LITERALS IN screen.py. A string that is not
listed here has no picture in the APK, and the screen draws a hollow box where the words
should be. Keeping them together means the render tool and the screen cannot disagree:
the tool renders exactly this list.

NUMBERS ARE NOT IN IT. Counts, times and file names change every second and cannot be
pre-rendered; they are composed from the DIGITS set, one glyph at a time, which is safe
for Latin digits and punctuation in a way it never is for Thai.
"""

#: Composed per character at draw time.
#:
#: Digits, punctuation AND THE LATIN ALPHABET. None of it has contextual shaping, so
#: blitting one glyph after another is honest for all of it -- which is never true of Thai.
#:
#: The letters are here for one reason: a fault message. When the model throws, what comes
#: back is "OrtException: ORT_INVALID_ARGUMENT" or the like, written by somebody else at
#: the moment it happens, and no pre-rendered word list can contain it. Without these the
#: phone could only say "โมเดลผิดพลาด" and the cause stayed on a developer's machine --
#: which is exactly how a wrong tensor shape survived two builds.
# The em dash is in here and not in the word list because it is used as a VALUE: it is
# what stands in the 'ต้องการ' column when a record had no target, exactly as the bench
# shows it, and values are composed a glyph at a time.
DIGITS = ("0123456789:/.,-+ x%—"
          "()[]_'\"!?#=<>*"
          "abcdefghijklmnopqrstuvwxyz"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

#: Every fixed string, grouped by where it appears. The groups are for reading; the tool
#: renders the union.
COUNTING = [
    "DrugCount",
    "ms",
    "วินาที",
    "เม็ด",
    "ภาพจากกล้อง",
    "บันทึกไว้",
    "รีเฟรช",
    "กำลังเตรียมระบบ",
    "พร้อมนับ",
    "ยังไม่กำหนดจำนวน",
    "ครบตามจำนวน",
    "เกิน",
    "ขาด",
    "จำนวนที่ต้องการ",
    "กำหนดกรอบนับ",
    "กำหนดกรอบใหม่",
    "ยกเลิก",
    "ล้างกรอบ",
    "บันทึกผล",
    "ดูรายการที่บันทึก",
    "เกินจำนวนที่ต้องการ",
    "นำออกก่อนจึงบันทึกได้",
    "ภาพจากกล้องหยุด",
    "ตรวจกล้อง",
    "ภาพค้าง",
    "เฉพาะในกรอบ",
    "นับทั้งภาพ",
    "กำลังเปิดกล้อง",
    "บันทึกแล้ว",
    "กำหนดกรอบแล้ว",
    "กรอบเล็กเกินไป",
    # Placing the counting region a corner at a time. The tray is a quadrilateral whenever
    # the lens is not square to the bench, which is always, so the rubber band and its
    # "ลากนิ้วคลุมพื้นที่ถาด" went together.
    "แตะมุมถาดทีละมุม",
    "อีก",
    "จุด",
    "ถอยจุด",
    "พลิกภาพ",
    "พลิกภาพซ้าย-ขวาแล้ว",
    "เลิกพลิกภาพแล้ว",
    "มุมนี้แคบเกินไป",
    "แตะให้ห่างจากมุมอื่น",
    "โมเดลผิดพลาด",
    # Multi-round counting. A tray holds about sixty tablets before they start lying on
    # one another, so a prescription for a hundred is physically two pours, and these are
    # the words that let the screen say so. Every one of them has to be in this list or it
    # ships as a hollow box -- which is the whole reason this file exists.
    "เก็บรอบที่",
    "นับใหม่",
    "กดอีกครั้ง",
    "จะทิ้งยอดสะสม",
    "กดอีกครั้งเพื่อเริ่มนับใหม่",
    "เริ่มนับใหม่",
    "ทิ้งยอดสะสม",
    "แล้ว",
    "เก็บแล้ว",
    "รอบ",
    "รวม",
    "ในถาด",
    "รอกวาดถาด",
    "กวาดเม็ดในถาดออกให้หมด",
    "แล้วจึงเทรอบต่อไป",
    "ถาดว่างแล้ว",
    "เทรอบต่อไปได้",
    "ถาดว่าง",
    "ยังไม่มีอะไรให้เก็บ",
    "ตัวเลขยังไม่นิ่ง",
    "รอสักครู่",
    "สะสมแล้ว",
    "บันทึกว่าไม่ครบ",
    "ยังเทค้างอยู่",
    "ยอดสะสมจะถูกล้าง",
    "แตะอีกครั้งเพื่อบันทึก",
]

RECORDS = [
    "รายการที่บันทึก",
    "บันทึกทั้งหมด",
    "ครบตามจำนวน",
    "ไม่ตรงจำนวน",
    "ทั้งหมด",
    "วันนี้",
    "ไม่ตรงจำนวน",
    "ช่วงวันเวลา",
    "ตั้งแต่",
    "ถึง",
    "นับได้",
    "ต้องการ",
    "ผล",
    "ครบ",
    "ไม่ได้ตั้ง",
    "รายการ",
    "ยังไม่มีรายการที่บันทึก",
    "ไม่มีรายการที่ตรงกับตัวกรอง",
    "ส่งออก CSV",
    "ส่งออกแล้ว",
    "ล้างข้อมูลทั้งหมด",
    "ลบแล้ว",
    "ย้อนกลับ",
    "ปิด",
    "เวลาโมเดล",
    "เวลา",
    "รีเฟรช",
    "(นับทั้งวัน)",
    "ภาพขณะบันทึก",
    "เลือกรายการเพื่อดูภาพ",
    "ไม่มีภาพ",
    "อ่านรายการใหม่แล้ว",
    "เจอทั้งหมด",
    "ลบรายการนี้",
    "ยืนยันลบทั้งหมด",
    "แตะอีกครั้งเพื่อยืนยัน",
    "ช่วงวันที่",
    "ใส่จำนวน",
    "ตกลง",
    "ลบ",
    "ล้าง",
    "เลือกวันที่",
    "วันนี้",
    "ล้างช่วง",
]

#: Weekday initials for the calendar, Monday first -- the week a Thai calendar starts on.
WEEKDAYS = ["จ", "อ", "พ", "พฤ", "ศ", "ส", "อา"]

#: The months, for a date that is read rather than typed.
MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
          "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

#: What the render tool draws. Sorted and de-duplicated so the asset list is stable
#: between runs and a diff of it means something changed on screen.
ALL = sorted(set(COUNTING + RECORDS + MONTHS + WEEKDAYS))
