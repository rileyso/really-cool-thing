from nicegui import ui


# -----------------------------
# PAGE
# -----------------------------

ui.label('Indoor Visual Navigator').classes(
    'text-3xl font-bold mb-4'
)


with ui.row().classes('w-full gap-6'):

    # LEFT SIDE
    with ui.card().classes('w-[45%] p-4'):

        ui.label('Query Image').classes(
            'text-xl font-semibold'
        )

        image_display = ui.image(
            'https://placehold.co/600x400?text=Upload+Image'
        ).classes(
            'w-full h-[400px] object-contain'
        )

        ui.upload(
            label='Upload Image'
        ).classes(
            'w-full'
        )

        ui.separator()

        detected_room = ui.label(
            'Detected Room: ---'
        ).classes(
            'text-lg'
        )

        confidence = ui.label(
            'Confidence: ---'
        )


    # RIGHT SIDE
    with ui.card().classes('w-[50%] p-4'):

        ui.label('Building Map').classes(
            'text-xl font-semibold'
        )

        map_display = ui.image(
            'https://placehold.co/800x500?text=Floorplan'
        ).classes(
            'w-full h-[500px] object-contain'
        )


# -----------------------------
# NAVIGATION CONTROLS
# -----------------------------

with ui.card().classes('w-full mt-6 p-4'):

    ui.label('Navigation').classes(
        'text-xl font-semibold'
    )

    with ui.row().classes('items-center gap-4'):

        destination = ui.select(
            [
                'Room 401',
                'Room 402',
                'Room 403',
                'Room 415',
            ],
            label='Destination'
        ).classes(
            'w-64'
        )

        ui.button(
            'Navigate'
        )

        ui.button(
            'Clear Route'
        ).props(
            'outline'
        )


ui.run(
    title='Indoor Visual Navigator'
)