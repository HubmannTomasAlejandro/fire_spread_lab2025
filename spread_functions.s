	.file	"spread_functions.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.p2align 4
	.globl	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf
	.type	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf, @function
_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf:
.LFB10551:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	vxorps	%xmm7, %xmm7, %xmm7
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rcx, %rbp
	subq	$120, %rsp
	.cfi_def_cfa_offset 176
	movswl	(%rdi), %ecx
	movq	%rbp, %rax
	xorl	%r14d, %r14d
	vcvtsd2ss	176(%rsp), %xmm7, %xmm5
	vmovsd	232(%rsp), %xmm7
	vmovsd	200(%rsp), %xmm4
	vmovss	%xmm1, 88(%rsp)
	vmovsd	192(%rsp), %xmm6
	vmovss	%xmm0, 84(%rsp)
	vmovss	%xmm2, 92(%rsp)
	movl	$1, %r13d
	vmovss	%xmm3, 104(%rsp)
	leaq	_ZL6ANGLES(%rip), %r12
	movq	%rdx, %rbx
	movq	%rsi, %rbp
	vmovsd	240(%rsp), %xmm1
	movl	%ecx, 80(%rsp)
	vmovss	%xmm5, 108(%rsp)
	movq	%rax, %r15
	vmovsd	%xmm7, 40(%rsp)
	vmovsd	184(%rsp), %xmm7
	vmovsd	%xmm4, 72(%rsp)
	vcvtss2sd	%xmm5, %xmm5, %xmm4
	vmovsd	%xmm6, 64(%rsp)
	vmovsd	%xmm1, 48(%rsp)
	vmovsd	%xmm7, 56(%rsp)
	movq	%rdi, 96(%rsp)
	jmp	.L7
	.p2align 4
	.p2align 3
.L2:
	cmpb	$2, %sil
	je	.L15
	cmpb	$3, %sil
	je	.L16
	vxorps	%xmm2, %xmm2, %xmm2
	cmpb	$4, %sil
	jne	.L17
.L6:
	vmovss	%xmm2, (%rbx,%r14,4)
	incq	%r14
	addq	$20, %rbp
	cmpq	$8, %r14
	je	.L18
.L7:
	movzbl	9(%rbp), %esi
	cmpb	$1, %sil
	jne	.L2
	vaddsd	208(%rsp), %xmm4, %xmm3
	vmovss	12(%rbp), %xmm6
	vmovss	16(%rbp), %xmm1
	vcvtsd2ss	%xmm3, %xmm3, %xmm3
.L3:
	shlx	%r14, %r13, %rsi
	andq	(%r15), %rsi
	vxorps	%xmm2, %xmm2, %xmm2
	je	.L6
	movswl	0(%rbp), %esi
	movl	80(%rsp), %eax
	vxorps	%xmm5, %xmm5, %xmm5
	vmovsd	%xmm4, 32(%rsp)
	vcvtss2sd	104(%rsp), %xmm2, %xmm2
	vmovss	%xmm6, 28(%rsp)
	vmovss	%xmm3, 20(%rsp)
	vmovss	%xmm1, 24(%rsp)
	movl	%esi, %edi
	movl	%esi, 16(%rsp)
	subl	%eax, %edi
	vcvtsi2ssl	%edi, %xmm5, %xmm0
	vdivss	84(%rsp), %xmm0, %xmm0
	vmovsd	%xmm2, 8(%rsp)
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	atan@PLT
	call	sin@PLT
	movq	96(%rsp), %rax
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm1
	vmovss	(%r12,%r14,4), %xmm0
	vsubss	4(%rax), %xmm0, %xmm0
	vmovsd	%xmm1, (%rsp)
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	cos@PLT
	movl	16(%rsp), %esi
	vxorps	%xmm5, %xmm5, %xmm5
	vmovapd	%xmm0, %xmm7
	vmovss	28(%rsp), %xmm6
	vmovss	20(%rsp), %xmm3
	vcvtsd2ss	%xmm7, %xmm7, %xmm7
	vmovsd	(%rsp), %xmm1
	vcvtsi2ssl	%esi, %xmm5, %xmm0
	vsubss	88(%rsp), %xmm0, %xmm0
	vcvtss2sd	%xmm7, %xmm7, %xmm7
	vmulsd	56(%rsp), %xmm7, %xmm7
	vdivss	92(%rsp), %xmm0, %xmm0
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vcvtss2sd	%xmm6, %xmm6, %xmm6
	vfmadd132sd	40(%rsp), %xmm3, %xmm6
	vcvtsd2ss	%xmm6, %xmm6, %xmm6
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vfmadd231sd	64(%rsp), %xmm0, %xmm7
	vcvtss2sd	%xmm6, %xmm6, %xmm6
	vfmadd231sd	72(%rsp), %xmm1, %xmm7
	vmovss	24(%rsp), %xmm1
	vcvtss2sd	%xmm1, %xmm1, %xmm0
	vfmadd132sd	48(%rsp), %xmm6, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm7, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vxorps	.LC1(%rip), %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	call	exp@PLT
	vaddsd	.LC2(%rip), %xmm0, %xmm0
	vmovsd	8(%rsp), %xmm2
	vmovsd	32(%rsp), %xmm4
	vdivsd	%xmm0, %xmm2, %xmm2
	vcvtsd2ss	%xmm2, %xmm2, %xmm2
	jmp	.L6
	.p2align 4
	.p2align 3
.L15:
	vaddsd	216(%rsp), %xmm4, %xmm3
	vmovss	12(%rbp), %xmm6
	vmovss	16(%rbp), %xmm1
	vcvtsd2ss	%xmm3, %xmm3, %xmm3
	jmp	.L3
	.p2align 4
	.p2align 3
.L16:
	vaddsd	224(%rsp), %xmm4, %xmm3
	vmovss	12(%rbp), %xmm6
	vmovss	16(%rbp), %xmm1
	vcvtsd2ss	%xmm3, %xmm3, %xmm3
	jmp	.L3
	.p2align 4
	.p2align 3
.L18:
	addq	$120, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L17:
	.cfi_restore_state
	vmovss	12(%rbp), %xmm6
	vmovss	16(%rbp), %xmm1
	vmovss	108(%rsp), %xmm3
	jmp	.L3
	.cfi_endproc
.LFE10551:
	.size	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf, .-_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf
	.section	.text._ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,"axG",@progbits,_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.type	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, @function
_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv:
.LFB11329:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L24
	movq	32(%rbx), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	$0, (%rbx)
	movl	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movl	$0, 24(%rbx)
	movq	$0, 32(%rbx)
.L24:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11329:
	.size	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, .-_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC3:
	.string	"vector::_M_realloc_append"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB10:
	.text
.LHOTB10:
	.p2align 4
	.globl	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff:
.LFB10552:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA10552
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	movq	%rdx, %r14
	subq	$640, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	8(%rsi), %rax
	movq	%r10, -600(%rbp)
	movq	%rdi, -624(%rbp)
	movq	%rsi, -504(%rbp)
	vmovss	%xmm0, -580(%rbp)
	vmovss	%xmm1, -584(%rbp)
	vmovss	%xmm2, -588(%rbp)
	vmovss	%xmm3, -592(%rbp)
	movq	%rax, -512(%rbp)
	movq	8(%rdx), %rax
	movq	(%rsi), %r10
	movq	%rax, -472(%rbp)
	movq	(%rdx), %rsi
	movq	$0, -568(%rbp)
	movq	$0, -552(%rbp)
	movq	$0, -528(%rbp)
	subq	%rsi, %rax
	movq	%rsi, -480(%rbp)
	movq	%rax, %r15
	sarq	$4, %r15
	movq	%r15, -536(%rbp)
	testq	%rax, %rax
	je	.L27
	xorl	%r13d, %r13d
	jmp	.L37
	.p2align 6
	.p2align 4
	.p2align 3
.L187:
	vmovdqu	(%rsi), %xmm0
	addq	$16, %rax
	incq	%r13
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -552(%rbp)
	cmpq	%r15, %r13
	jnb	.L27
.L188:
	movq	(%r14), %rsi
.L37:
	movq	%r13, %rax
	movq	-568(%rbp), %rdi
	salq	$4, %rax
	addq	%rax, %rsi
	movq	-552(%rbp), %rax
	cmpq	%rdi, %rax
	jne	.L187
	movq	-528(%rbp), %rax
	movq	%rdi, %r12
	movabsq	$576460752303423487, %rdi
	subq	%rax, %r12
	movq	%r12, %rax
	sarq	$4, %rax
	cmpq	%rdi, %rax
	je	.L177
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L32
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rbx
.L33:
	movq	%rbx, %rdi
	movq	%rsi, -464(%rbp)
	movq	%r10, -456(%rbp)
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	movq	-464(%rbp), %rsi
	movq	-552(%rbp), %rcx
	movq	-528(%rbp), %rdx
	movq	-456(%rbp), %r10
	vmovdqu	(%rsi), %xmm0
	cmpq	%rdx, %rcx
	vmovdqu	%xmm0, (%rax,%r12)
	je	.L120
	movq	%rax, %rsi
	.p2align 5
	.p2align 4
	.p2align 3
.L35:
	vmovdqu	(%rdx), %xmm0
	addq	$16, %rdx
	addq	$16, %rsi
	vmovdqu	%xmm0, -16(%rsi)
	cmpq	%rdx, %rcx
	jne	.L35
	movq	-552(%rbp), %rdx
	movq	-528(%rbp), %rdi
	subq	%rdi, %rdx
	addq	%rax, %rdx
.L34:
	leaq	16(%rdx), %rdi
	movq	%rdi, -552(%rbp)
	movq	-528(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L36
	movq	-568(%rbp), %rsi
	movq	%rax, -464(%rbp)
	movq	%r10, -456(%rbp)
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	-464(%rbp), %rax
	movq	-456(%rbp), %r10
.L36:
	leaq	(%rax,%rbx), %rdi
	incq	%r13
	movq	%rax, -528(%rbp)
	movq	%rdi, -568(%rbp)
	cmpq	%r15, %r13
	jb	.L188
.L27:
	movl	$8, %edi
	movq	%r10, -456(%rbp)
.LEHB1:
	call	_Znwm@PLT
.LEHE1:
	movq	%r15, (%rax)
	movq	-456(%rbp), %r10
	movq	%rax, -616(%rbp)
	addq	$8, %rax
	movq	%rax, -576(%rbp)
	movl	$0, -312(%rbp)
	movq	-512(%rbp), %rax
	vmovq	%r10, %xmm7
	vpinsrq	$1, %rax, %xmm7, %xmm6
	imulq	%r10, %rax
	vmovdqa	%xmm6, -672(%rbp)
	vmovdqa	%xmm6, -336(%rbp)
	movq	%rax, %rbx
	testq	%rax, %rax
	jne	.L189
	movq	$0, -456(%rbp)
	movq	$0, -648(%rbp)
	movl	$0, -628(%rbp)
	movq	$0, -640(%rbp)
.L38:
	movq	-480(%rbp), %rsi
	cmpq	%rsi, -472(%rbp)
	je	.L48
	movq	(%r14), %rcx
	xorl	%esi, %esi
	movl	$1, %r8d
	.p2align 4
	.p2align 3
.L49:
	movq	8(%rcx), %rax
	movq	-456(%rbp), %rdi
	imulq	%r10, %rax
	addq	(%rcx), %rax
	leaq	63(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$6, %rdx
	leaq	(%rdi,%rdx,8), %rdi
	cqto
	shrq	$58, %rdx
	addq	%rdx, %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L190
	incq	%rsi
	shlx	%rax, %r8, %rax
	addq	$16, %rcx
	orq	%rax, (%rdi)
	cmpq	%r15, %rsi
	jb	.L49
.L48:
	movq	-576(%rbp), %rax
	movq	$0, -496(%rbp)
	movl	$8, %r12d
	movq	%rax, -608(%rbp)
	cmpq	$-1, %r15
	je	.L43
	.p2align 4
	.p2align 3
.L42:
	movq	-536(%rbp), %rax
	leaq	-432(%rbp), %rdi
	movq	%rdi, -520(%rbp)
	movq	%rax, -560(%rbp)
	cmpq	%rax, -496(%rbp)
	jnb	.L191
	.p2align 4
	.p2align 3
.L70:
	movq	-496(%rbp), %rax
	movq	%r10, -464(%rbp)
	movq	-528(%rbp), %rsi
	movq	-520(%rbp), %rdi
	salq	$4, %rax
	addq	%rsi, %rax
	movq	-504(%rbp), %rsi
	movq	(%rax), %r12
	movq	8(%rax), %rbx
	movq	%r12, %rdx
	movq	%rbx, %rcx
	vzeroupper
.LEHB2:
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	leaq	-208(%rbp), %rdi
	movl	%r12d, %eax
	movq	-464(%rbp), %r10
	movl	%ebx, %r11d
	movq	%rdi, -544(%rbp)
	movq	%rdi, %r14
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	leaq	-272(%rbp), %r8
	movl	$1, %r9d
	jmp	.L55
	.p2align 4
	.p2align 3
.L50:
	vmovdqa	_ZL18out_of_bounds_cell(%rip), %xmm0
	movl	16+_ZL18out_of_bounds_cell(%rip), %edx
	movl	%edx, -384(%rbp)
	movl	%edx, 16(%r14)
	vmovdqa	%xmm0, -400(%rbp)
	vmovdqu	%xmm0, (%r14)
.L53:
	shlx	%r12, %r9, %rdx
	incq	%r12
	addq	$20, %r14
	andn	%r15, %rdx, %r15
	cmpq	$8, %r12
	je	.L192
.L55:
	leaq	_ZL5MOVES(%rip), %rdi
	movl	(%rdi,%r12,8), %r13d
	leaq	4(%rdi), %rsi
	movl	(%rsi,%r12,8), %ebx
	addl	%eax, %r13d
	addl	%r11d, %ebx
	movl	%r13d, %ecx
	movl	%r13d, (%r8,%r12,4)
	orl	%ebx, %ecx
	movl	%ebx, 32(%r8,%r12,4)
	js	.L50
	cmpl	%r10d, %r13d
	jge	.L50
	movl	-512(%rbp), %esi
	cmpl	%esi, %ebx
	jge	.L50
	movq	-504(%rbp), %rsi
	movslq	%r13d, %r13
	movslq	%ebx, %rbx
	leaq	-400(%rbp), %rdi
	movq	%r13, %rdx
	movq	%rbx, %rcx
	movq	%r8, -488(%rbp)
	movl	%r11d, -480(%rbp)
	movl	%eax, -472(%rbp)
	movq	%r10, -464(%rbp)
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	movq	-464(%rbp), %r10
	movl	-384(%rbp), %edx
	movl	$1, %r9d
	movq	-456(%rbp), %rax
	vmovdqa	-400(%rbp), %xmm0
	movl	-480(%rbp), %r11d
	movq	-488(%rbp), %r8
	imulq	%r10, %rbx
	movl	%edx, 16(%r14)
	leaq	63(%r13,%rbx), %rdx
	addq	%r13, %rbx
	cmovns	%rbx, %rdx
	sarq	$6, %rdx
	vmovdqu	%xmm0, (%r14)
	leaq	(%rax,%rdx,8), %rcx
	movq	%rbx, %rdx
	movl	-472(%rbp), %eax
	sarq	$63, %rdx
	shrq	$58, %rdx
	addq	%rdx, %rbx
	andl	$63, %ebx
	subq	%rdx, %rbx
	jns	.L52
	addq	$64, %rbx
	subq	$8, %rcx
.L52:
	shlx	%rbx, %r9, %rbx
	andq	(%rcx), %rbx
	shlx	%r12, %r9, %rdx
	jne	.L53
	cmpb	$0, 8(%r14)
	je	.L53
	incq	%r12
	orq	%rdx, %r15
	addq	$20, %r14
	cmpq	$8, %r12
	jne	.L55
.L192:
	testq	%r15, %r15
	jne	.L193
.L69:
	incq	-496(%rbp)
	movq	-536(%rbp), %rsi
	movq	-496(%rbp), %rax
	cmpq	%rsi, %rax
	jne	.L70
	movq	-576(%rbp), %rax
	movq	-608(%rbp), %rdi
	cmpq	%rdi, %rax
	je	.L72
	movq	-560(%rbp), %rdi
	addq	$8, %rax
	movq	%rdi, -8(%rax)
	movq	%rax, -576(%rbp)
	movq	%rsi, %rax
.L74:
	movq	%rax, -496(%rbp)
	movq	-560(%rbp), %rsi
	cmpq	%rsi, %rax
	je	.L73
	movq	%rsi, -536(%rbp)
	jmp	.L42
	.p2align 4
	.p2align 3
.L193:
	movq	-600(%rbp), %rax
	subq	$80, %rsp
	movq	%r10, -464(%rbp)
	movq	%r8, -472(%rbp)
	movq	%r15, -440(%rbp)
	leaq	-440(%rbp), %rcx
	leaq	-368(%rbp), %rdx
	vmovdqu	(%rax), %ymm0
	vmovdqu	%ymm0, (%rsp)
	vmovdqu	32(%rax), %ymm0
	vmovdqu	%ymm0, 32(%rsp)
	movq	64(%rax), %rax
	movq	%rax, 64(%rsp)
	vmovss	-588(%rbp), %xmm2
	vmovss	-584(%rbp), %xmm1
	vmovss	-580(%rbp), %xmm0
	vmovss	-592(%rbp), %xmm3
	movq	-544(%rbp), %rsi
	movq	-520(%rbp), %rdi
	vzeroupper
	call	_Z18spread_probabilityRK4CellPS0_16SimulationParamsfffPfRSt6bitsetILm8EEf
	vmovdqa	rng(%rip), %ymm1
	movl	$65535, %eax
	addq	$80, %rsp
	vbroadcastss	.LC6(%rip), %ymm2
	movq	-464(%rbp), %r10
	movl	$1, %r9d
	vpslld	$13, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vpsrld	$17, %ymm0, %ymm1
	vpxor	%ymm0, %ymm1, %ymm1
	vpslld	$5, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vmovd	%eax, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vmovdqa	%ymm0, rng(%rip)
	vpand	%ymm1, %ymm0, %ymm1
	vpsrld	$16, %ymm0, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vcvtdq2ps	%ymm1, %ymm1
	vfmadd132ps	%ymm2, %ymm1, %ymm0
	vbroadcastss	.LC8(%rip), %ymm1
	vmulps	%ymm1, %ymm0, %ymm0
	vcmpps	$17, -368(%rbp), %ymm0, %ymm0
	vmovmskps	%ymm0, %r13d
	testb	%r13b, %r13b
	je	.L69
	movq	-472(%rbp), %r8
	movzbl	%r13b, %r13d
	xorl	%r12d, %r12d
	jmp	.L68
	.p2align 4
	.p2align 4
	.p2align 3
.L57:
	incq	%r12
	cmpq	$8, %r12
	je	.L69
.L68:
	btl	%r12d, %r13d
	jnc	.L57
	movq	-552(%rbp), %rax
	movq	-568(%rbp), %rdi
	movslq	(%r8,%r12,4), %r14
	movslq	32(%r8,%r12,4), %rbx
	cmpq	%rdi, %rax
	je	.L58
	vmovq	%r14, %xmm4
	addq	$16, %rax
	vpinsrq	$1, %rbx, %xmm4, %xmm0
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -552(%rbp)
.L59:
	imulq	%r10, %rbx
	movq	-456(%rbp), %rdi
	leaq	63(%r14,%rbx), %rax
	addq	%r14, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	leaq	(%rdi,%rax,8), %rdx
	movq	%rbx, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbx
	andl	$63, %ebx
	subq	%rax, %rbx
	jns	.L67
	addq	$64, %rbx
	subq	$8, %rdx
.L67:
	shlx	%rbx, %r9, %rbx
	incq	-560(%rbp)
	orq	%rbx, (%rdx)
	jmp	.L57
.L58:
	movq	-528(%rbp), %rax
	movq	%rdi, %rcx
	subq	%rax, %rcx
	movabsq	$576460752303423487, %rax
	movq	%rcx, %rdx
	sarq	$4, %rdx
	cmpq	%rax, %rdx
	je	.L178
	testq	%rdx, %rdx
	movl	$1, %eax
	cmovne	%rdx, %rax
	addq	%rdx, %rax
	jc	.L62
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	salq	$4, %rax
	movq	%rax, -464(%rbp)
	movq	%rax, %rdi
.L63:
	movq	%r8, -488(%rbp)
	movq	%rcx, -480(%rbp)
	movq	%r10, -472(%rbp)
	vzeroupper
	call	_Znwm@PLT
	movq	-480(%rbp), %rcx
	vmovq	%r14, %xmm5
	movq	%rax, %r15
	movq	-472(%rbp), %r10
	vpinsrq	$1, %rbx, %xmm5, %xmm0
	movq	-488(%rbp), %r8
	movl	$1, %r9d
	vmovdqu	%xmm0, (%rax,%rcx)
	movq	-552(%rbp), %rcx
	movq	-528(%rbp), %rax
	cmpq	%rax, %rcx
	je	.L124
	movq	%r15, %rdx
	.p2align 5
	.p2align 4
	.p2align 3
.L65:
	vmovdqu	(%rax), %xmm0
	addq	$16, %rax
	addq	$16, %rdx
	vmovdqu	%xmm0, -16(%rdx)
	cmpq	%rax, %rcx
	jne	.L65
	movq	-552(%rbp), %rax
	movq	-528(%rbp), %rsi
	subq	%rsi, %rax
	addq	%r15, %rax
.L64:
	addq	$16, %rax
	movq	%rax, -552(%rbp)
	movq	-528(%rbp), %rax
	testq	%rax, %rax
	je	.L66
	movq	-568(%rbp), %rsi
	movq	%rax, %rdi
	movq	%r8, -480(%rbp)
	movq	%r10, -472(%rbp)
	subq	%rax, %rsi
	call	_ZdlPvm@PLT
	movq	-480(%rbp), %r8
	movl	$1, %r9d
	movq	-472(%rbp), %r10
.L66:
	movq	-464(%rbp), %rax
	movq	%r15, -528(%rbp)
	addq	%r15, %rax
	movq	%rax, -568(%rbp)
	jmp	.L59
.L190:
	addq	$64, %rax
	incq	%rsi
	addq	$16, %rcx
	shlx	%rax, %r8, %rax
	orq	%rax, -8(%rdi)
	cmpq	%r15, %rsi
	jb	.L49
	jmp	.L48
.L72:
	movq	-616(%rbp), %rax
	movq	-576(%rbp), %r14
	movabsq	$1152921504606846975, %rdx
	subq	%rax, %r14
	movq	%r14, %rax
	sarq	$3, %rax
	cmpq	%rdx, %rax
	je	.L179
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L77
	movabsq	$1152921504606846975, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	leaq	0(,%rax,8), %rbx
.L78:
	movq	%r10, -464(%rbp)
	movq	%rbx, %rdi
	vzeroupper
	call	_Znwm@PLT
.LEHE2:
	movq	%rax, %r12
	movq	-560(%rbp), %rax
	testq	%r14, %r14
	movq	-464(%rbp), %r10
	leaq	8(%r14), %r13
	movq	%rax, (%r12,%r14)
	jg	.L194
	cmpq	$0, -616(%rbp)
	leaq	(%r12,%r13), %rax
	movq	%rax, -576(%rbp)
	jne	.L80
.L81:
	leaq	(%r12,%rbx), %rax
	movq	%r12, -616(%rbp)
	movq	%rax, -608(%rbp)
	movq	-536(%rbp), %rax
	jmp	.L74
.L191:
	movq	-576(%rbp), %rax
	movq	-608(%rbp), %rsi
	cmpq	%rsi, %rax
	je	.L72
	movq	-536(%rbp), %rsi
	addq	$8, %rax
	movq	%rsi, -8(%rax)
	movq	%rax, -576(%rbp)
.L73:
	movq	-608(%rbp), %r12
	movq	-616(%rbp), %rax
	subq	%rax, %r12
.L43:
	vbroadcasti128	-672(%rbp), %ymm0
	movq	-624(%rbp), %rax
	movq	-648(%rbp), %rbx
	movl	-628(%rbp), %r13d
	movq	$0, 32(%rax)
	movl	$0, 40(%rax)
	movq	$0, 48(%rax)
	movl	$0, 56(%rax)
	movq	$0, 64(%rax)
	vmovdqu	%ymm0, (%rax)
	movq	-456(%rbp), %rax
	subq	%rax, %rbx
	leaq	0(%r13,%rbx,8), %r14
	testq	%r14, %r14
	jne	.L83
	xorl	%ecx, %ecx
	vzeroupper
.L84:
	movl	-628(%rbp), %eax
	movq	-648(%rbp), %rdi
	addq	%rbx, %rcx
	xorl	%edx, %edx
	movl	$1, %r8d
	testl	%eax, %eax
	jne	.L88
	jmp	.L95
	.p2align 6
	.p2align 4
	.p2align 3
.L195:
	incl	%edx
	decq	%r13
	je	.L95
.L88:
	movq	(%rcx), %rsi
	movq	(%rdi), %r10
	shlx	%rdx, %r8, %rax
	andq	%rax, %r10
	movq	%rsi, %r9
	orq	%rax, %r9
	andn	%rsi, %rax, %rax
	testq	%r10, %r10
	cmovne	%r9, %rax
	movq	%rax, (%rcx)
	cmpl	$63, %edx
	jne	.L195
	addq	$8, %rdi
	addq	$8, %rcx
	xorl	%edx, %edx
	decq	%r13
	jne	.L88
.L95:
	movq	-624(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-552(%rbp), %rbx
	vmovdqu	%xmm0, 72(%rax)
	movq	$0, 88(%rax)
	movq	-528(%rbp), %rax
	subq	%rax, %rbx
	je	.L196
	movabsq	$9223372036854775792, %rax
	cmpq	%rbx, %rax
	jb	.L197
	movq	%rbx, %rdi
.LEHB3:
	call	_Znwm@PLT
.LEHE3:
	movq	%rax, %rcx
.L90:
	movq	-624(%rbp), %rax
	addq	%rcx, %rbx
	movq	-528(%rbp), %rsi
	movq	%rbx, 88(%rax)
	movq	-552(%rbp), %rbx
	movq	%rcx, 72(%rax)
	cmpq	%rsi, %rbx
	je	.L99
	subq	%rsi, %rbx
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	call	memcpy@PLT
	movq	%rax, %rcx
	addq	%rbx, %rcx
.L99:
	movq	-624(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-576(%rbp), %rbx
	movq	%rcx, 80(%rax)
	vmovdqu	%xmm0, 96(%rax)
	movq	$0, 112(%rax)
	movq	-616(%rbp), %rax
	subq	%rax, %rbx
	je	.L100
	movabsq	$9223372036854775800, %rax
	cmpq	%rbx, %rax
	jb	.L198
	movq	%rbx, %rdi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rdi
	vmovq	%rax, %xmm7
	movq	-624(%rbp), %rax
	vpunpcklqdq	%xmm7, %xmm7, %xmm0
	leaq	(%rdi,%rbx), %r13
	vmovdqu	%xmm0, 96(%rax)
	movq	%r13, 112(%rax)
	cmpq	$8, %rbx
	jle	.L199
	movq	%rbx, %rdx
	movq	-616(%rbp), %rsi
	call	memcpy@PLT
	cmpq	$0, -456(%rbp)
	movq	-624(%rbp), %rax
	movq	%r13, 104(%rax)
	je	.L109
.L108:
	movq	-640(%rbp), %rsi
	movq	-456(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L109:
	movq	-616(%rbp), %rdi
	movq	%r12, %rsi
	call	_ZdlPvm@PLT
.L110:
	movq	-528(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L26
	movq	-568(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L26:
	movq	-624(%rbp), %rax
	leaq	-48(%rbp), %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L194:
	.cfi_restore_state
	movq	-616(%rbp), %rsi
	movq	%r14, %rdx
	movq	%r12, %rdi
	call	memcpy@PLT
	movq	-464(%rbp), %r10
	leaq	(%r12,%r13), %rax
	movq	%rax, -576(%rbp)
.L80:
	movq	-608(%rbp), %rsi
	movq	%r10, -464(%rbp)
	movq	-616(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	-464(%rbp), %r10
	jmp	.L81
.L196:
	xorl	%ecx, %ecx
	jmp	.L90
.L100:
	cmpq	$0, -456(%rbp)
	movq	-624(%rbp), %rax
	movq	$0, 112(%rax)
	je	.L105
	movq	-640(%rbp), %rsi
	movq	-456(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L105:
	cmpq	$0, -616(%rbp)
	je	.L110
	jmp	.L109
	.p2align 4
	.p2align 3
.L124:
	movq	%r15, %rax
	jmp	.L64
.L83:
	leaq	63(%r14), %r15
	movq	%r15, %rdx
	shrq	$6, %rdx
	salq	$3, %rdx
	movq	%rdx, %rdi
	movq	%rdx, -464(%rbp)
	vzeroupper
.LEHB5:
	call	_Znwm@PLT
.LEHE5:
	movq	-464(%rbp), %rdx
	movq	%rax, %rcx
	addq	%rax, %rdx
	movq	-624(%rbp), %rax
	testq	%r14, %r14
	movq	%rdx, 64(%rax)
	movq	%rcx, 32(%rax)
	movl	$0, 40(%rax)
	movq	%r14, %rdx
	movq	%r15, %rax
	cmovns	%r14, %rax
	sarq	$63, %rdx
	sarq	$6, %rax
	shrq	$58, %rdx
	leaq	(%rcx,%rax,8), %rsi
	leaq	(%r14,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L200
.L85:
	movq	-624(%rbp), %rdi
	movq	%rsi, 48(%rdi)
	movl	%eax, 56(%rdi)
	cmpq	$8, %rbx
	jle	.L86
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	movq	-456(%rbp), %rsi
	call	memcpy@PLT
	movq	%rax, %rcx
	jmp	.L84
.L189:
	leaq	63(%rax), %rdx
	movq	%r10, -488(%rbp)
	movq	%rdx, %r13
	movq	%rdx, -464(%rbp)
	shrq	$6, %r13
	salq	$3, %r13
	movq	%r13, %rdi
.LEHB6:
	call	_Znwm@PLT
.LEHE6:
	movq	%rax, %rdi
	movq	%rax, -456(%rbp)
	leaq	(%rax,%r13), %rax
	movq	-488(%rbp), %r10
	movq	%rax, -640(%rbp)
	movq	-464(%rbp), %rax
	testq	%rbx, %rbx
	movq	%rbx, %rdx
	cmovns	%rbx, %rax
	sarq	$63, %rdx
	sarq	$6, %rax
	shrq	$58, %rdx
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -648(%rbp)
	leaq	(%rbx,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L39
.L186:
	movq	%r13, %rdx
	xorl	%esi, %esi
	movq	%r10, -464(%rbp)
	movl	%eax, -628(%rbp)
	movl	$0, -312(%rbp)
	call	memset@PLT
	movq	-464(%rbp), %r10
	jmp	.L38
.L39:
	subq	$8, -648(%rbp)
	addl	$64, %eax
	jmp	.L186
.L200:
	addq	$64, %rax
	subq	$8, %rsi
	jmp	.L85
.L120:
	movq	%rax, %rdx
	jmp	.L34
.L197:
.LEHB7:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE7:
.L199:
	movq	-616(%rbp), %rax
	cmpq	$0, -456(%rbp)
	movq	(%rax), %rax
	movq	%rax, (%rdi)
	movq	-624(%rbp), %rax
	movq	%r13, 104(%rax)
	jne	.L108
	jmp	.L109
.L86:
	jne	.L84
	movq	-456(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rcx)
	jmp	.L84
.L77:
	movabsq	$9223372036854775800, %rbx
	jmp	.L78
	.p2align 4
	.p2align 3
.L198:
.LEHB8:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE8:
.L32:
	movabsq	$9223372036854775792, %rbx
	jmp	.L33
.L62:
	movabsq	$9223372036854775792, %rax
	movq	%rax, -464(%rbp)
	movq	%rax, %rdi
	jmp	.L63
.L134:
	movq	%rax, %rbx
	jmp	.L96
.L131:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L114
.L173:
	jmp	.L174
.L135:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L116
.L132:
	movq	%rax, %rbx
	jmp	.L112
.L133:
	movq	%rax, %rbx
	jmp	.L44
.L175:
	jmp	.L176
	.section	.gcc_except_table,"a",@progbits
.LLSDA10552:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE10552-.LLSDACSB10552
.LLSDACSB10552:
	.uleb128 .LEHB0-.LFB10552
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L173-.LFB10552
	.uleb128 0
	.uleb128 .LEHB1-.LFB10552
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L135-.LFB10552
	.uleb128 0
	.uleb128 .LEHB2-.LFB10552
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L175-.LFB10552
	.uleb128 0
	.uleb128 .LEHB3-.LFB10552
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L131-.LFB10552
	.uleb128 0
	.uleb128 .LEHB4-.LFB10552
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L132-.LFB10552
	.uleb128 0
	.uleb128 .LEHB5-.LFB10552
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L134-.LFB10552
	.uleb128 0
	.uleb128 .LEHB6-.LFB10552
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L133-.LFB10552
	.uleb128 0
	.uleb128 .LEHB7-.LFB10552
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L131-.LFB10552
	.uleb128 0
	.uleb128 .LEHB8-.LFB10552
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L132-.LFB10552
	.uleb128 0
.LLSDACSE10552:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC10552
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold:
.LFSB10552:
.L96:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	movq	-624(%rbp), %rdi
	addq	$32, %rdi
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
.L97:
	movq	-456(%rbp), %rax
	leaq	-320(%rbp), %rdi
	movq	%rax, -320(%rbp)
	movq	-648(%rbp), %rax
	movq	%rax, -304(%rbp)
	movl	-628(%rbp), %eax
	movl	%eax, -296(%rbp)
	movq	-640(%rbp), %rax
	movq	%rax, -288(%rbp)
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movq	-608(%rbp), %rsi
	movq	-616(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L116
.L45:
	movq	-616(%rbp), %rdi
	call	_ZdlPvm@PLT
.L116:
	movq	-568(%rbp), %rsi
	movq	-528(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L117
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L117:
	movq	%rbx, %rdi
.LEHB9:
	call	_Unwind_Resume@PLT
.LEHE9:
.L112:
	movq	-624(%rbp), %rax
	movq	72(%rax), %rdi
	movq	88(%rax), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L184
	vzeroupper
	call	_ZdlPvm@PLT
.L114:
	movq	-624(%rbp), %rdi
	addq	$32, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L97
.L179:
	leaq	.LC3(%rip), %rdi
	vzeroupper
.LEHB10:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE10:
.L130:
.L174:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L116
.L136:
.L176:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L97
.L177:
	leaq	.LC3(%rip), %rdi
.LEHB11:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE11:
.L44:
	xorl	%edx, %edx
	xorl	%esi, %esi
	xorl	%ecx, %ecx
	leaq	-320(%rbp), %rdi
	movq	%rsi, -288(%rbp)
	movq	%rdx, -320(%rbp)
	movq	%rdx, -304(%rbp)
	movl	%ecx, -296(%rbp)
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movl	$8, %esi
	jmp	.L45
.L184:
	vzeroupper
	jmp	.L114
.L178:
	leaq	.LC3(%rip), %rdi
	vzeroupper
.LEHB12:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE12:
	.cfi_endproc
.LFE10552:
	.section	.gcc_except_table
.LLSDAC10552:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC10552-.LLSDACSBC10552
.LLSDACSBC10552:
	.uleb128 .LEHB9-.LCOLDB10
	.uleb128 .LEHE9-.LEHB9
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB10-.LCOLDB10
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L136-.LCOLDB10
	.uleb128 0
	.uleb128 .LEHB11-.LCOLDB10
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L130-.LCOLDB10
	.uleb128 0
	.uleb128 .LEHB12-.LCOLDB10
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L136-.LCOLDB10
	.uleb128 0
.LLSDACSEC10552:
	.section	.text.unlikely
	.text
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.section	.text.unlikely
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold
.LCOLDE10:
	.text
.LHOTE10:
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I_rng, @function
_GLOBAL__sub_I_rng:
.LFB11634:
	.cfi_startproc
	vmovdqa	.LC11(%rip), %ymm0
	vmovdqa	%ymm0, rng(%rip)
	vzeroupper
	ret
	.cfi_endproc
.LFE11634:
	.size	_GLOBAL__sub_I_rng, .-_GLOBAL__sub_I_rng
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_rng
	.globl	rng
	.bss
	.align 32
	.type	rng, @object
	.size	rng, 32
rng:
	.zero	32
	.section	.rodata
	.align 16
	.type	_ZL18out_of_bounds_cell, @object
	.size	_ZL18out_of_bounds_cell, 20
_ZL18out_of_bounds_cell:
	.value	0
	.zero	2
	.long	0
	.byte	0
	.byte	4
	.zero	2
	.long	0
	.long	0
	.align 32
	.type	_ZL5MOVES, @object
	.size	_ZL5MOVES, 64
_ZL5MOVES:
	.long	-1
	.long	-1
	.long	-1
	.long	0
	.long	-1
	.long	1
	.long	0
	.long	-1
	.long	0
	.long	1
	.long	1
	.long	-1
	.long	1
	.long	0
	.long	1
	.long	1
	.align 32
	.type	_ZL6ANGLES, @object
	.size	_ZL6ANGLES, 32
_ZL6ANGLES:
	.long	1075235812
	.long	1078530011
	.long	1081824209
	.long	1070141403
	.long	1083624420
	.long	1061752795
	.long	0
	.long	1085271519
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC2:
	.long	0
	.long	1072693248
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC6:
	.long	1199570944
	.align 4
.LC8:
	.long	796917760
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC11:
	.long	12345
	.long	67890
	.long	13579
	.long	24680
	.long	11223
	.long	44556
	.long	77889
	.long	99000
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.globl	__gxx_personality_v0
	.ident	"GCC: (Debian 14.2.0-16) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
