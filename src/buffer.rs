use crate::{
	command::CommandPool,
	device::{Device, Queue, SubmitFuture},
};
use ash::{version::DeviceV1_0, vk};
use std::{marker::PhantomData, mem::size_of, slice, sync::Arc};
use typenum::{Bit, B1};
use vk::BufferUsageFlags;
use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

pub struct Buffer<T: ?Sized> {
	device: Arc<Device>,
	pub vk: vk::Buffer,
	alloc: Allocation,
	size: u64,
	phantom: PhantomData<T>,
}
impl<T: ?Sized> Buffer<T> {
	pub fn size(&self) -> u64 {
		self.size
	}
}
impl<T> Buffer<[T]> {
	pub fn init_slice<CPU: Bit>(
		device: Arc<Device>,
		len: usize,
		_cpu: CPU,
		usage: BufferUsageFlags,
	) -> BufferInit<[T], CPU> {
		let size = size_of::<T>() as u64 * len as u64;

		let ci = ash::vk::BufferCreateInfo::builder().size(size).usage(usage).build();

		let usage = if CPU::BOOL { MemoryUsage::CpuOnly } else { MemoryUsage::GpuOnly };
		let aci = AllocationCreateInfo { usage, ..Default::default() };

		let (vk, alloc, _) = device.allocator.create_buffer(&ci, &aci).unwrap();

		let buf = Arc::new(Self { device, vk, alloc, size, phantom: PhantomData });
		BufferInit::new(buf)
	}

	pub fn len(&self) -> u64 {
		self.size / size_of::<T>() as u64
	}
}
impl<T: ?Sized> Drop for Buffer<T> {
	fn drop(&mut self) {
		unsafe { self.device.vk.destroy_buffer(self.vk, None) };
		self.device.allocator.free_memory(&self.alloc).unwrap();
	}
}
impl<T: ?Sized> BufferAbstract for Buffer<T> {
	fn vk(&self) -> vk::Buffer {
		self.vk
	}
}

pub struct BufferInit<T: ?Sized, CPU> {
	buf: Arc<Buffer<T>>,
	phantom: PhantomData<CPU>,
}
impl<T: ?Sized, CPU> BufferInit<T, CPU> {
	fn new(buf: Arc<Buffer<T>>) -> Self {
		Self { buf, phantom: PhantomData }
	}
}
impl<T: Send + Sync + 'static, CPU> BufferInit<[T], CPU> {
	pub fn copy_from_buffer(
		self,
		queue: &Arc<Queue>,
		pool: &Arc<CommandPool>,
		buffer: Arc<Buffer<[T]>>,
	) -> (Arc<Buffer<[T]>>, SubmitFuture) {
		let cmd = pool.record(true, false).copy_buffer(buffer, self.buf.clone()).build();
		let future = queue.submit(cmd);
		(self.buf, future)
	}
}
impl<T: Copy + 'static> BufferInit<[T], B1> {
	pub fn copy_from_slice(self, data: &[T]) -> Arc<Buffer<[T]>> {
		let buf = self.buf;
		let allocator = &buf.device.allocator;
		let alloc = &buf.alloc;

		let bufdata = allocator.map_memory(&alloc).unwrap();
		let bufdata = unsafe { slice::from_raw_parts_mut(bufdata as *mut T, (buf.size / size_of::<T>() as u64) as _) };
		bufdata.copy_from_slice(data);
		allocator.unmap_memory(&alloc).unwrap();

		buf
	}
}

pub trait BufferAbstract {
	fn vk(&self) -> vk::Buffer;
}
