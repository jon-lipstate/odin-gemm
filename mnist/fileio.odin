package scratch
import "core:os"
import "core:io"
import "core:encoding/endian"

//
read_labels :: proc(filename: string, allocator := context.allocator) -> []u8 {
	context.allocator = allocator
	contents, ok := os.read_entire_file_from_filename(filename, allocator)
	defer delete(contents)
	magic, mok := endian.get_u32(contents[:4], .Big)
	assert(magic == 2049)
	n_labels, lok := endian.get_u32(contents[4:8], .Big)
	// fmt.println("Labels:", n_labels)
	labels := make([]u8, n_labels)
	k := 0
	for i := 8; i < int(n_labels) + 8; i += 1 {
		labels[k] = contents[i]
		k += 1
	}
	return labels
}
read_images :: proc(img_path: string, label_path: string, allocator := context.allocator) -> []MNIST_Image {
	context.allocator = allocator
	labels := read_labels(label_path, allocator)
	defer delete(labels)

	contents, ok := os.read_entire_file_from_filename(img_path, allocator)
	magic, mok := endian.get_u32(contents[:4], .Big)
	assert(magic == 2051)
	n_images, lok := endian.get_u32(contents[4:8], .Big)
	n_rows, rok := endian.get_u32(contents[8:12], .Big)
	n_cols, cok := endian.get_u32(contents[12:16], .Big)
	assert(n_rows == n_cols && n_rows == 28)

	images := make([]MNIST_Image, n_images)
	k := 0
	for i := 16; i < int(n_images) + 16; i += 784 {
		copy(images[k].data[:], contents[i:i + 784])
		images[k].expected = labels[k]
		k += 1
	}
	delete(contents)
	return images
}
